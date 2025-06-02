import os
import shutil
import subprocess
from typing import List, Tuple, Dict, Any, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import gettempdir
from tqdm import tqdm
from joblib import Parallel, delayed

import datamol as dm
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdb as pdb

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import PandasTools
from openbabel import pybel, openbabel

try:
    from openeye import oechem
    from openeye import oeomega
    OPENEYE_AVAILABLE = True
except ImportError:
    OPENEYE_AVAILABLE = False

from .tools.Protein_Preparation import ProteinPreparation_Chimera
from .tools.tools import pybel_converter, pybel_flagger
from .tools.Ligand_Preparation import LigandPreparator
from .tools.Fix_Mol2 import fix_docking_poses_from_mol2

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Plants_Docking:
    def __init__(self, workdir: Path, pdb_ID: Path, crystal_path: Path, ligands_sdf: Path, 
                protonation_method: str = "cdp", tautomer_score_threshold: Optional[float] = None):
        """
        Initialize the Plants docking pipeline
        
        Args:
            workdir: Working directory for docking
            pdb_ID: Path to the PDB file
            crystal_path: Path to the crystal ligand file
            ligands_sdf: Path to the ligands SDF file
            protonation_method: Method for protonating ligands ("cdp", "oe", or "scrubber")
            tautomer_score_threshold: Score threshold for tautomer selection (None = best only, value = list within threshold)
        """
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True)

        self.pdb_ID = pdb_ID
        self.crystal_path = crystal_path
        self.ligands_sdf: Path = ligands_sdf
        self.ligands_mol2: List[Path] = []
        self.protein_cleaned = None
        self.protein_prepared = None
        self.binding_site: Tuple[str, str] = ("", "")

        self.docked_output_plants: Path = workdir / "output_plants"
        self.docked_output_plants.mkdir(exist_ok=True)

        self.docked_final_dir: Path = workdir / "output"
        self.docked_final_dir.mkdir(exist_ok=True)

        self.docked_plants: List[Path] = []
        
        # Store template SMILES mapping for MOL2 fixing
        self.template_smiles_mapping: Dict[str, str] = {}
        
        # Set the protonation method for ligand preparation
        if protonation_method.lower() not in ["cdp", "oe", "scrubber"]:
            raise ValueError(f"Protonation method must be 'cdp', 'oe', or 'scrubber', got {protonation_method}")
        
        self.protonation_method = protonation_method.lower()
        self.tautomer_score_threshold = tautomer_score_threshold

        self._plants_env_variable()

    def _plants_env_variable(self):
        os.environ['PATH'] = '/home/hitesit/Software/PLANTS:' + os.environ.get('PATH', '')

    def _source_macro(self):
        # Grab the protein from the PDB
        protein_path = self.pdb_ID
        reader = pdb.PDBFile.read(protein_path)
        struct_array = reader.get_structure(model=1)

        # Remove all and not protein
        macro_array = struct_array[struc.filter_amino_acids(struct_array)]

        protein_cleaned: Path = self.workdir / f"{self.pdb_ID.stem}_clean.pdb"
        self.protein_cleaned = protein_cleaned

        strucio.save_structure(str(protein_cleaned), macro_array)

    def _prepare_protein(self):
        pdb_mol2 = self.workdir / f"{self.pdb_ID.stem}_prep.mol2"
        chimera_prep = ProteinPreparation_Chimera()
        chimera_prep(self.protein_cleaned, pdb_mol2)

        self.protein_prepared = pdb_mol2

    def _define_binding_site(self):
        # Convert the crystal to mol2 and save it in the temp directory
        tmp_crystal: Path = Path(gettempdir()) / self.crystal_path.name
        pybel_converter(str(self.crystal_path), "sdf", str(tmp_crystal), "mol2")

        # Run the PLANTS command
        plants_command = f"plants.64bit --mode bind {str(tmp_crystal.absolute())} {str(self.protein_prepared.absolute())}"
        plants_results = subprocess.run(plants_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, cwd=self.workdir)
        box_infos = plants_results.stdout.split("\n")

        binding_site_center = box_infos[-4]
        binding_site_radius = box_infos[-3]

        self.binding_site = (binding_site_center, binding_site_radius)

    def _convert_to_mol2(self):
        ligands_mol2_folder = self.workdir / "ligands_mol2"
        ligands_mol2_folder.mkdir(exist_ok=True)
        
        # Initialize the ligand preparator with appropriate settings
        logger.info(f"Preparing ligands using {self.protonation_method} protonation method")
        
        preparator = LigandPreparator(
            protonation_method=self.protonation_method,
            enumerate_stereo=True,
            tautomer_score_threshold=self.tautomer_score_threshold,
            generate_3d=True
        )
        
        # Prepare molecules from SDF
        prepared_mols = preparator.prepare_from_sdf(self.ligands_sdf)
        
        # Store template SMILES mapping before saving to MOL2
        # Use the original molecules from the SDF file for more stable templates
        logger.info(f"Storing template SMILES mapping for MOL2 fixing")
        original_supplier = Chem.SDMolSupplier(str(self.ligands_sdf))
        
        # Create a mapping from original molecules to their SMILES
        original_smiles_map = {}
        for mol in original_supplier:
            if mol is not None:
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{len(original_smiles_map)}"
                # Use the original molecule SMILES (before protonation/preparation)
                mol_smiles = Chem.MolToSmiles(mol)
                original_smiles_map[mol_name] = mol_smiles
                
        # Map prepared molecules back to original SMILES
        for mol in prepared_mols:
            if mol is not None:
                mol_name = mol.GetProp("_Name")
                # Extract base name (before _Iso suffix)
                base_name = mol_name.split("_Iso")[0]  # e.g., "Apigenin_Iso0" -> "Apigenin"
                if base_name in original_smiles_map:
                    self.template_smiles_mapping[mol_name] = original_smiles_map[base_name]
                    logger.debug(f"Mapped {mol_name} to original SMILES: {original_smiles_map[base_name]}")
                else:
                    # Fallback to prepared molecule SMILES
                    mol_smiles = Chem.MolToSmiles(mol)
                    self.template_smiles_mapping[mol_name] = mol_smiles
                    logger.warning(f"No original SMILES found for {base_name}, using prepared: {mol_smiles}")
        
        # Save to MOL2 format for Plants
        logger.info(f"Converting prepared ligands to mol2 format")
        mol2_paths = preparator.save_to_mol2(prepared_mols, ligands_mol2_folder)
        
        # Add paths to the list
        self.ligands_mol2 = mol2_paths

    def write_conf(self, ligand_to_dock: Path, n_confs) -> Path:
        # Define the binding site
        center, radius = self.binding_site

        # Write the configuration file
        conf_path: Path = self.workdir / f"{ligand_to_dock.stem}_conf.txt"

        # Make some relpaths
        protein_prepared_REL: Path = self.protein_prepared.relative_to(self.workdir)
        ligand_to_dock_REL: Path = ligand_to_dock.relative_to(self.workdir)

        # Just the tmpdir for the output of Plants
        tmp_dir: Path = (self.docked_output_plants / ligand_to_dock.stem).relative_to(self.workdir)

        config_str = f"""### PLANTS configuration file
# scoring function and search settings
scoring_function chemplp
search_speed speed1

# input
protein_file {str(protein_prepared_REL)}
ligand_file {str(ligand_to_dock_REL)}

# output
output_dir {str(tmp_dir)}
write_multi_mol2 1

# binding site definition
{center}
{radius}

# cluster algorithm
cluster_structures {n_confs}
cluster_rmsd 1.0
"""
        with open(conf_path, "w") as f:
            f.write(config_str)

        return conf_path

    def _save_to_sdf(self, df: pd.DataFrame, name: str):
        docked_final_sdf = self.docked_final_dir / f"{name}_Plants.sdf"
        docked_final_sdf = docked_final_sdf.absolute()
        PandasTools.WriteSDF(df, str(docked_final_sdf),
                             idName="LIGAND_ENTRY", molColName="Molecule",
                             properties=list(df.columns))

    def main(self, n_confs: int, n_cpus: int):
        self._source_macro()
        self._prepare_protein()
        self._define_binding_site()
        self._convert_to_mol2()

        # Write the conf file and add the path to the list
        conf_files: List[Path] = []
        docked_plants: List[Path] = []
        for ligand in self.ligands_mol2:
            # Save the path of the future plants output
            lig_stem = ligand.stem
            plants_mol2 = self.docked_output_plants / lig_stem / "docked_ligands.mol2"

            self.docked_plants.append(plants_mol2)
            docked_plants.append(plants_mol2)

            # Execute the conf writing and add the path to the list
            conf_path = self.write_conf(ligand, n_confs).relative_to(self.workdir)
            conf_files.append(conf_path)

        def runner(conf_file: Path, docked_plants: Path) -> Tuple[pd.DataFrame, str]:
            try:
                command = f"plants.64bit --mode screen {str(conf_file)}"
                logger.info(f"Running: {command}")
                subprocess.run(command, shell=True, check=True, cwd=self.workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Check if the docked_plants file exists
                if not docked_plants.exists():
                    logger.warning(f"Docking output file not found: {docked_plants}")
                    # Return empty dataframe with ligand name
                    return (pd.DataFrame(), docked_plants.parent.name)

                converter = Convert_Plants(docked_plants, self.template_smiles_mapping)
                comb_df = converter.main()
                return (comb_df, docked_plants.parent.name)
            except Exception as e:
                # Log the error but don't fail the entire process
                logger.error(f"Error processing ligand {docked_plants.parent.name}: {str(e)}")
                # Return empty dataframe with ligand name
                return (pd.DataFrame(), docked_plants.parent.name)

        # Process ligands with two options
        if n_cpus > 1:
            # Option 1: Parallel processing with immediate saving of results
            def process_and_save(conf_file, docked_plant):
                df, name = runner(conf_file, docked_plant)
                if not df.empty:
                    # Create a copy of self._save_to_sdf that can be called here
                    sdf_path = self.docked_final_dir / f"{name}_Plants.sdf"
                    logger.info(f"Saving results for {name} to {sdf_path}")
                    PandasTools.WriteSDF(df, str(sdf_path.absolute()),
                                        idName="LIGAND_ENTRY", molColName="Molecule",
                                        properties=list(df.columns))
                return (df, name)  # Still return results for counting/reporting
                
            results = Parallel(n_jobs=n_cpus)(
                delayed(process_and_save)(conf_file, docked_plant) 
                for conf_file, docked_plant in zip(conf_files, docked_plants)
            )
            
            # Count successes and failures
            success_count = sum(1 for df, _ in results if not df.empty)
            logger.info(f"Docking completed: {success_count} successful out of {len(conf_files)} ligands")
        else:
            # Option 2: Sequential processing with immediate saving of results
            success_count = 0
            for conf_file, docked_plant in zip(conf_files, docked_plants):
                df, name = runner(conf_file, docked_plant)
                if not df.empty:
                    logger.info(f"Saving results for {name}")
                    self._save_to_sdf(df, name)
                    success_count += 1
                else:
                    logger.warning(f"Skipping empty results for {name}")
            
            logger.info(f"Docking completed: {success_count} successful out of {len(conf_files)} ligands")


class Convert_Plants:
    def __init__(self, plants_mol: Path, template_smiles_mapping: Optional[Dict[str, str]] = None):
        self.plants_mol = plants_mol
        self.plants_dir: Path = self.plants_mol.parent
        self.template_smiles_mapping = template_smiles_mapping or {}

    def get_rdmol_df(self):
        # Try normal sanitized reading first
        try:
            logger.debug(f"Attempting normal MOL2 reading with sanitization for {self.plants_mol}")
            rd_mols: List = dm.read_mol2file(self.plants_mol, sanitize=True)
            logger.debug(f"Successfully read {len(rd_mols)} molecules with normal sanitization")
            
            # Check if we actually got valid molecules
            valid_mols = [mol for mol in rd_mols if mol is not None]
            if len(valid_mols) == 0:
                raise Exception("All molecules failed sanitization")
            elif len(valid_mols) < len(rd_mols):
                logger.warning(f"Some molecules failed sanitization: {len(valid_mols)}/{len(rd_mols)} valid")
                rd_mols = valid_mols
            
        except Exception as e:
            logger.warning(f"Normal MOL2 reading failed for {self.plants_mol}: {e}")
            logger.info(f"Attempting Fix_Mol2 fallback strategy for {self.plants_mol}...")
            
            try:
                logger.info(f"=== ENTERING Fix_Mol2 fallback strategy ===")
                # Get molecule name from directory name (e.g., "Apigenin_Iso0")
                mol_name = self.plants_dir.name
                logger.info(f"Processing molecule: {mol_name}")
                template_smiles = self.template_smiles_mapping.get(mol_name)
                
                if template_smiles is None:
                    logger.error(f"No template SMILES found for molecule {mol_name}")
                    logger.error(f"Available templates: {list(self.template_smiles_mapping.keys())}")
                    logger.info(f"Skipping Fix_Mol2 and going directly to Pybel pure conversion fallback...")
                    # Don't return here, let it fall through to pybel fallback
                    raise Exception("No template SMILES available for Fix_Mol2 strategy")
                    
                logger.info(f"Found template SMILES for {mol_name}: {template_smiles[:50]}...")
                
                # Read molecules without sanitization
                broken_mols = dm.read_mol2file(self.plants_mol, sanitize=False)
                logger.info(f"Read {len(broken_mols)} unsanitized molecules from {self.plants_mol}")
                
                # Apply Fix_Mol2 strategy
                logger.info(f"Applying Fix_Mol2 strategy using template SMILES: {template_smiles}")
                logger.info(f"MOL2 file path: {self.plants_mol}")
                logger.info(f"First broken mol atoms: {broken_mols[0].GetNumAtoms() if broken_mols else 'No molecules'}")
                rd_mols = fix_docking_poses_from_mol2(broken_mols, template_smiles, verbose=False)  # Disable verbose for cleaner output
                
                # Filter out None results and log details
                valid_mols = [mol for mol in rd_mols if mol is not None]
                logger.info(f"Fix_Mol2 strategy: {len(valid_mols)} fixed out of {len(broken_mols)} input molecules")
                if len(valid_mols) != len(rd_mols):
                    logger.warning(f"Fix_Mol2 returned {len(rd_mols) - len(valid_mols)} None molecules")
                rd_mols = valid_mols
                
                if not rd_mols:
                    logger.error(f"Fix_Mol2 strategy failed to fix any molecules for {self.plants_mol}")
                    raise Exception("Fix_Mol2 strategy could not fix any molecules")
                    
            except Exception as fix_error:
                logger.error(f"Fix_Mol2 fallback strategy also failed for {self.plants_mol}: {fix_error}")
                logger.info(f"Attempting Pybel pure conversion fallback for {self.plants_mol}...")
                
                try:
                    logger.info(f"=== ENTERING Pybel pure conversion fallback ===")
                    # Read MOL2 file using OpenBabel/pybel
                    pybel_mols = list(pybel.readfile("mol2", str(self.plants_mol)))
                    logger.info(f"Pybel successfully read {len(pybel_mols)} molecules from {self.plants_mol}")
                    
                    # Convert each pybel molecule to RDKit via SDF intermediate
                    rd_mols = []
                    for i, pybel_mol in enumerate(pybel_mols):
                        try:
                            # Convert pybel molecule to SDF string
                            sdf_string = pybel_mol.write("sdf")
                            
                            # Create RDKit molecule from SDF (no sanitization to avoid validation issues)
                            rdkit_mol = Chem.MolFromMolBlock(sdf_string, sanitize=False)
                            
                            if rdkit_mol is not None:
                                # Use directory name for molecule identification (standardized naming like RxDock)
                                mol_base_name = self.plants_dir.name  # e.g., "Lig_Complex_3_Iso0_Taut0"
                                pose_name = f"{mol_base_name}_Plants-P{i+1}"
                                rdkit_mol.SetProp("_Name", pose_name)
                                rd_mols.append(rdkit_mol)
                                logger.debug(f"Successfully converted pose {i+1}: {pose_name}")
                            else:
                                logger.warning(f"Failed to create RDKit molecule from pybel molecule {i+1}")
                                
                        except Exception as mol_error:
                            logger.warning(f"Error converting pybel molecule {i+1}: {mol_error}")
                            continue
                    
                    logger.info(f"Pybel pure conversion: {len(rd_mols)} successfully converted out of {len(pybel_mols)} molecules")
                    
                    if not rd_mols:
                        logger.error(f"Pybel pure conversion failed to convert any molecules for {self.plants_mol}")
                        return pd.DataFrame()
                        
                except Exception as pybel_error:
                    logger.error(f"Pybel pure conversion fallback also failed for {self.plants_mol}: {pybel_error}")
                    return pd.DataFrame()

        rows = []
        for rd_mol in rd_mols:
            if rd_mol is not None:  # Filter out None molecules
                row = {
                    "LIGAND_ENTRY": rd_mol.GetProp("_Name"),
                    "Molecule": rd_mol
                }
                rows.append(row)
        
        logger.info(f"Created {len(rows)} rows for DataFrame from {len(rd_mols)} molecules")
        return pd.DataFrame(rows)

    def retrieve_csv(self):
        score_csv: Path = self.plants_dir / "ranking.csv"
        score_df = pd.read_csv(score_csv)

        return score_df

    def main(self):
        """Process docked ligands and merge with scores."""
        # Load molecules from mol2 file
        rdmol_df = self.get_rdmol_df()
        
        # Load scores from CSV
        score_df = self.retrieve_csv()
        
        # For debugging
        logger.debug(f"Molecule names in mol2: {rdmol_df['LIGAND_ENTRY'].tolist()}")
        logger.debug(f"Entry names in CSV: {score_df['LIGAND_ENTRY'].tolist()}")
        
        # The key insight: When using OpenEye, the molecule names don't match PLANTS output naming convention
        # PLANTS uses names like "Lig_1_Iso0_entry_00001_conf_01" 
        # But OpenEye uses names like "Lig_1_S0" which don't match
        
        # Check if we have a naming mismatch
        if not rdmol_df.empty and not score_df.empty:
            # Get the base name of the directory which is our molecule ID
            mol_base_name = self.plants_dir.name  # e.g., "Lig_1_Iso0" or "Lig_1_S0"
            
            # Check if any CSV entries contain this name
            matching_entries = [entry for entry in score_df['LIGAND_ENTRY'].tolist() 
                              if mol_base_name.split('_')[0] in entry and 
                                 mol_base_name.split('_')[1] in entry]
            
            if matching_entries:
                logger.debug(f"Found matching entries in CSV: {matching_entries}")
                
                # Modify rdmol_df to use the PLANTS naming for merging
                for i, row in rdmol_df.iterrows():
                    # Find the corresponding entry in the score_df based on index order
                    if i < len(matching_entries):
                        rdmol_df.at[i, 'LIGAND_ENTRY'] = matching_entries[i]
        
        # Merge the dataframes
        try:
            # Check if both dataframes have data and the required column
            if rdmol_df.empty or score_df.empty:
                logger.warning(f"Cannot merge: rdmol_df empty={rdmol_df.empty}, score_df empty={score_df.empty}")
                return pd.DataFrame()
            
            if "LIGAND_ENTRY" not in rdmol_df.columns:
                logger.error(f"LIGAND_ENTRY column missing from molecule dataframe")
                return pd.DataFrame()
                
            if "LIGAND_ENTRY" not in score_df.columns:
                logger.error(f"LIGAND_ENTRY column missing from score dataframe")
                return pd.DataFrame()
            
            # Try standard merge
            comb_df = pd.merge(rdmol_df, score_df, on="LIGAND_ENTRY")
        except Exception as e:
            logger.warning(f"Merge failed: {str(e)}")
            
            # If merge fails, create a new dataframe with molecules and add empty score columns
            comb_df = rdmol_df.copy()
            for col in score_df.columns:
                if col not in comb_df.columns and col != "LIGAND_ENTRY":
                    comb_df[col] = float('nan')
        
        # Process the final molecule names using standardized format (like RxDock)
        if not comb_df.empty:
            try:
                # Use the directory name as the base name (preserves individual ligand identity)
                mol_base_name = self.plants_dir.name  # e.g., "Lig_Complex_3_Iso0_Taut0"
                
                # Create standardized pose names: {base_name}_Plants-P{pose_number}
                standardized_names = []
                for i in range(len(comb_df)):
                    pose_name = f"{mol_base_name}_Plants-P{i+1}"
                    standardized_names.append(pose_name)
                
                comb_df["LIGAND_ENTRY"] = standardized_names
                
                # Update the molecule _Name properties to match the standardized names
                for i, (index, row) in enumerate(comb_df.iterrows()):
                    mol = row["Molecule"]
                    if mol is not None:
                        pose_name = standardized_names[i]
                        mol.SetProp("_Name", pose_name)
                        mol.SetProp("Software", "plants")  # Add software identifier
                
                logger.info(f"Applied standardized naming: {mol_base_name}_Plants-P1 to P{len(comb_df)}")
                
            except Exception as e:
                logger.warning(f"Error applying standardized names: {str(e)}")
                # Fallback to generic naming if something goes wrong
                mol_base_name = self.plants_dir.name if hasattr(self, 'plants_dir') else "Unknown"
                comb_df["LIGAND_ENTRY"] = [f"{mol_base_name}_Plants-P{i+1}" for i in range(len(comb_df))]
        
        return comb_df