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
        
        # Try using PandasTools first (works for sanitized molecules)
        try:
            PandasTools.WriteSDF(df, str(docked_final_sdf),
                                 idName="LIGAND_ENTRY", molColName="Molecule",
                                 properties=list(df.columns))
            logger.debug(f"Successfully saved SDF using PandasTools: {docked_final_sdf}")
        except Exception as e:
            logger.warning(f"PandasTools.WriteSDF failed for {name}: {e}")
            logger.info(f"Attempting manual SDF writing for unsanitized molecules...")
            
            # Fallback: manual SDF writing for unsanitized molecules
            try:
                written_count = self._write_unsanitized_sdf(df, docked_final_sdf)
                if written_count > 0:
                    logger.info(f"Successfully saved SDF using manual method: {docked_final_sdf} ({written_count} molecules)")
                else:
                    logger.error(f"Manual SDF writing failed: no molecules could be written for {name}")
                    logger.info(f"Attempting ultimate fallback: recreate from source using pybel conversion...")
                    
                    # Final fallback: use pybel to recreate the molecules and save to SDF
                    self._pybel_sdf_fallback(name, docked_final_sdf)
            except Exception as manual_error:
                logger.error(f"Manual SDF writing also failed for {name}: {manual_error}")
                logger.info(f"Attempting ultimate fallback: recreate from source using pybel conversion...")
                
                # Final fallback: use pybel to recreate the molecules and save to SDF
                self._pybel_sdf_fallback(name, docked_final_sdf)
    
    def _write_unsanitized_sdf(self, df: pd.DataFrame, output_path: Path):
        """
        Manually write SDF file for unsanitized molecules using RDKit SDWriter directly
        Returns the number of molecules successfully written
        """
        from rdkit import Chem
        
        written_count = 0
        with Chem.SDWriter(str(output_path)) as writer:
            for _, row in df.iterrows():
                mol = row["Molecule"]
                if mol is not None:
                    # Set molecule name
                    if "LIGAND_ENTRY" in row:
                        mol.SetProp("_Name", str(row["LIGAND_ENTRY"]))
                    
                    # Set all other properties from the dataframe
                    for col in df.columns:
                        if col != "Molecule" and not pd.isna(row[col]):
                            mol.SetProp(col, str(row[col]))
                    
                    # Write molecule (this should work even for unsanitized molecules)
                    try:
                        writer.write(mol)
                        written_count += 1
                    except Exception as mol_error:
                        logger.warning(f"Failed to write molecule {row.get('LIGAND_ENTRY', 'unknown')}: {mol_error}")
                        continue
        
        return written_count

    def _pybel_sdf_fallback(self, name: str, output_path: Path):
        """
        Ultimate fallback: use pybel to convert MOL2 directly to SDF.
        This preserves 3D coordinates and creates a valid SDF file even if RDKit fails.
        """
        try:
            logger.info(f"Starting pybel SDF fallback for {name}")
            
            # Find the corresponding MOL2 file
            mol2_file = None
            for docked_plant in self.docked_plants:
                if docked_plant.parent.name == name:
                    mol2_file = docked_plant
                    break
            
            if mol2_file is None or not mol2_file.exists():
                logger.error(f"Could not find MOL2 file for {name}")
                return
            
            logger.info(f"Converting {mol2_file} to SDF using pybel")
            
            # Read with pybel and convert to SDF
            pybel_mols = list(pybel.readfile("mol2", str(mol2_file)))
            if not pybel_mols:
                logger.error(f"Pybel could not read any molecules from {mol2_file}")
                return
            
            logger.info(f"Pybel loaded {len(pybel_mols)} molecules from {mol2_file}")
            
            # Write to SDF format
            with open(output_path, 'w') as f:
                for i, pybel_mol in enumerate(pybel_mols):
                    # Set molecule title/name
                    if hasattr(pybel_mol, 'title') and pybel_mol.title:
                        mol_name = pybel_mol.title
                    else:
                        mol_name = f"{name}_entry_{i+1:05d}_conf_{i+1:02d}"
                    
                    # Convert to SDF string
                    sdf_string = pybel_mol.write("sdf")
                    
                    # Modify the SDF string to add properties following the SDF field philosophy
                    lines = sdf_string.strip().split('\n')
                    if lines:
                        # Find the $$$$  line and add properties before it
                        insert_pos = len(lines)
                        for j, line in enumerate(lines):
                            if line.strip() == "$$$$":
                                insert_pos = j
                                break
                        
                        # Add properties
                        properties = [
                            f">  <LIGAND_ENTRY>",
                            f"{mol_name}_Plants-P{i+1}",
                            "",
                            f">  <Original_Name>",
                            f"{mol_name}",
                            "",
                            f">  <Conversion_Method>",
                            f"pybel_fallback",
                            "",
                            f">  <Note>",
                            f"Converted from broken MOL2 using pybel as ultimate fallback",
                            ""
                        ]
                        
                        # Insert properties
                        for prop in reversed(properties):
                            lines.insert(insert_pos, prop)
                        
                        # Write to file
                        f.write('\n'.join(lines) + '\n')
            
            logger.info(f"Pybel SDF fallback succeeded: saved {len(pybel_mols)} molecules to {output_path}")
            
        except Exception as e:
            logger.error(f"Pybel SDF fallback failed for {name}: {e}")
            # Create a minimal placeholder file to avoid completely empty output
            try:
                with open(output_path, 'w') as f:
                    f.write(f"# Failed to convert {name} - all conversion strategies exhausted\n")
                    f.write(f"# Error: {str(e)}\n")
                logger.warning(f"Created placeholder file for {name} - all conversion strategies failed")
            except:
                logger.error(f"Could not even create placeholder file for {name}")

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

                # Let Convert_Plants handle its own exceptions and fallback strategies
                logger.info(f"Creating Convert_Plants object for {docked_plants.parent.name}")
                converter = Convert_Plants(docked_plants, self.template_smiles_mapping)
                logger.info(f"Calling converter.main() for {docked_plants.parent.name}")
                comb_df = converter.main()
                logger.info(f"converter.main() completed for {docked_plants.parent.name}, got {len(comb_df)} rows")
                return (comb_df, docked_plants.parent.name)
            except subprocess.CalledProcessError as e:
                # PLANTS docking command failed
                logger.error(f"PLANTS docking failed for ligand {docked_plants.parent.name}: {str(e)}")
                return (pd.DataFrame(), docked_plants.parent.name)
            except Exception as e:
                # Only catch truly catastrophic errors that prevent basic processing
                logger.error(f"Unexpected error processing ligand {docked_plants.parent.name}: {str(e)}")
                # Even in this case, try to create the converter and let it handle the problem
                try:
                    if docked_plants.exists():
                        converter = Convert_Plants(docked_plants, self.template_smiles_mapping)
                        comb_df = converter.main()
                        return (comb_df, docked_plants.parent.name)
                except:
                    pass
                return (pd.DataFrame(), docked_plants.parent.name)

        # Process ligands with two options
        if n_cpus > 1:
            # Option 1: Parallel processing with immediate saving of results
            def process_and_save(conf_file, docked_plant):
                df, name = runner(conf_file, docked_plant)
                if not df.empty:
                    logger.info(f"Saving results for {name}")
                    self._save_to_sdf(df, name)
                else:
                    logger.warning(f"No valid results for {name}, skipping SDF file creation")
                    # Do NOT create any file if all fallback strategies failed
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
                    logger.warning(f"No valid results for {name}, skipping SDF file creation")
                    # Do NOT create any file if all fallback strategies failed
            
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
                    return pd.DataFrame()
                    
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
                    logger.warning(f"Fix_Mol2 strategy failed to fix any molecules for {self.plants_mol}")
                    logger.info(f"Attempting final fallback: using unsanitized molecules directly")
                    
                    # Final fallback: use the broken molecules directly (unsanitized)
                    try:
                        broken_mols = dm.read_mol2file(self.plants_mol, sanitize=False)
                        logger.info(f"Final fallback: Read {len(broken_mols)} unsanitized molecules")
                        
                        if broken_mols:
                            # Use the unsanitized molecules as-is
                            rd_mols = [mol for mol in broken_mols if mol is not None]
                            logger.warning(f"Using {len(rd_mols)} unsanitized molecules (bond orders may be incorrect)")
                            
                            # Check if we actually got valid molecules after filtering
                            if not rd_mols:
                                logger.error(f"All unsanitized molecules were None, trying pybel fallback")
                                logger.info(f"Attempting ultimate fallback: pybel conversion with DU atom fixing...")
                                
                                # Ultimate fallback: pybel conversion with DU atom fixing
                                rd_mols = self._pybel_fallback_strategy()
                                if not rd_mols:
                                    logger.error(f"All strategies including pybel fallback failed for {self.plants_mol}")
                                    return pd.DataFrame()
                        else:
                            logger.error(f"No molecules read from MOL2 file, trying pybel fallback")
                            logger.info(f"Attempting ultimate fallback: pybel conversion with DU atom fixing...")
                            
                            # Ultimate fallback: pybel conversion with DU atom fixing
                            rd_mols = self._pybel_fallback_strategy()
                            if not rd_mols:
                                logger.error(f"All strategies including pybel fallback failed for {self.plants_mol}")
                                return pd.DataFrame()
                            
                    except Exception as final_error:
                        logger.error(f"All fallback strategies failed for {self.plants_mol}: {final_error}")
                        logger.info(f"Attempting ultimate fallback: pybel conversion with DU atom fixing...")
                        
                        # Ultimate fallback: pybel conversion with DU atom fixing
                        rd_mols = self._pybel_fallback_strategy()
                        if not rd_mols:
                            logger.error(f"All strategies including pybel fallback failed for {self.plants_mol}")
                            return pd.DataFrame()
                    
            except Exception as fix_error:
                logger.warning(f"Fix_Mol2 fallback strategy failed for {self.plants_mol}: {fix_error}")
                logger.info(f"Attempting final fallback: using unsanitized molecules directly")
                
                # Final fallback: use the broken molecules directly (unsanitized)
                try:
                    broken_mols = dm.read_mol2file(self.plants_mol, sanitize=False)
                    logger.info(f"Final fallback: Read {len(broken_mols)} unsanitized molecules")
                    
                    if broken_mols:
                        # Use the unsanitized molecules as-is
                        rd_mols = [mol for mol in broken_mols if mol is not None]
                        logger.warning(f"Using {len(rd_mols)} unsanitized molecules (bond orders may be incorrect)")
                        
                        # Check if we actually got valid molecules after filtering
                        if not rd_mols:
                            logger.error(f"All unsanitized molecules were None, trying pybel fallback")
                            logger.info(f"Attempting ultimate fallback: pybel conversion with DU atom fixing...")
                            
                            # Ultimate fallback: pybel conversion with DU atom fixing
                            rd_mols = self._pybel_fallback_strategy()
                            if not rd_mols:
                                logger.error(f"All strategies including pybel fallback failed for {self.plants_mol}")
                                return pd.DataFrame()
                    else:
                        logger.error(f"No molecules read from MOL2 file, trying pybel fallback")
                        logger.info(f"Attempting ultimate fallback: pybel conversion with DU atom fixing...")
                        
                        # Ultimate fallback: pybel conversion with DU atom fixing
                        rd_mols = self._pybel_fallback_strategy()
                        if not rd_mols:
                            logger.error(f"All strategies including pybel fallback failed for {self.plants_mol}")
                            return pd.DataFrame()
                        
                except Exception as final_error:
                    logger.error(f"All fallback strategies failed for {self.plants_mol}: {final_error}")
                    logger.info(f"Attempting ultimate fallback: pybel conversion with DU atom fixing...")
                    
                    # Ultimate fallback: pybel conversion with DU atom fixing
                    rd_mols = self._pybel_fallback_strategy()
                    if not rd_mols:
                        logger.error(f"All strategies including pybel fallback failed for {self.plants_mol}")
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

    def _pybel_fallback_strategy(self) -> List[Chem.Mol]:
        """
        Ultimate fallback: Use pybel to convert MOL2 to SDF, then fix DU atoms using template SMILES.
        Prioritizes 3D coordinate preservation over chemical validity.
        """
        try:
            logger.info(f"Starting pybel fallback strategy for {self.plants_mol}")
            
            # Get molecule name and template SMILES
            mol_name = self.plants_dir.name
            template_smiles = self.template_smiles_mapping.get(mol_name)
            
            if template_smiles is None:
                logger.warning(f"No template SMILES found for {mol_name}, proceeding without DU fixing")
                template_mol = None
            else:
                logger.info(f"Using template SMILES: {template_smiles[:50]}...")
                template_mol = Chem.MolFromSmiles(template_smiles)
                if template_mol is None:
                    logger.warning(f"Invalid template SMILES, proceeding without DU fixing")
            
            # Read with pybel and convert to SDF format
            pybel_mols = pybel.readfile("mol2", str(self.plants_mol))
            fixed_mols = []
            
            for i, pybel_mol in enumerate(pybel_mols):
                logger.debug(f"Processing pybel molecule {i+1}")
                
                # Convert to SDF format string
                sdf_string = pybel_mol.write("sdf")
                
                # Read back with RDKit
                rd_mol = Chem.MolFromMolBlock(sdf_string, sanitize=False)
                if rd_mol is None:
                    logger.warning(f"Failed to convert pybel molecule {i+1} to RDKit")
                    continue
                
                # Fix DU atoms if template is available
                if template_mol is not None:
                    rd_mol = self._fix_du_atoms(rd_mol, template_mol, mol_name, i+1)
                
                # Set molecule name from original pybel molecule
                if hasattr(pybel_mol, 'title') and pybel_mol.title:
                    rd_mol.SetProp("_Name", pybel_mol.title)
                else:
                    rd_mol.SetProp("_Name", f"{mol_name}_conf_{i+1:02d}")
                
                fixed_mols.append(rd_mol)
            
            if fixed_mols:
                logger.info(f"Pybel fallback strategy succeeded: {len(fixed_mols)} molecules recovered")
                return fixed_mols
            else:
                logger.warning(f"Pybel fallback strategy failed: no valid molecules recovered")
                return []
                
        except Exception as e:
            logger.error(f"Pybel fallback strategy crashed: {e}")
            return []

    def _fix_du_atoms(self, mol: Chem.Mol, template_mol: Chem.Mol, mol_name: str, pose_num: int) -> Chem.Mol:
        """
        Fix DU (dummy) atoms by inferring correct atom types from template SMILES.
        Preserves 3D coordinates as much as possible.
        """
        try:
            logger.debug(f"Fixing DU atoms for {mol_name} pose {pose_num}")
            
            # Check if there are any DU atoms
            du_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "Du"]
            if not du_atoms:
                logger.debug(f"No DU atoms found in {mol_name} pose {pose_num}")
                return mol
            
            logger.info(f"Found {len(du_atoms)} DU atoms in {mol_name} pose {pose_num}")
            
            # If atom counts don't match, can't reliably fix
            if mol.GetNumAtoms() != template_mol.GetNumAtoms():
                logger.warning(f"Atom count mismatch: mol={mol.GetNumAtoms()}, template={template_mol.GetNumAtoms()}")
                logger.warning(f"Attempting to fix DU atoms without full matching")
                
                # Try to fix DU atoms by inferring from neighbors
                mol_copy = Chem.RWMol(mol)
                for atom in du_atoms:
                    # Get neighboring atoms
                    neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
                    neighbor_symbols = [n.GetSymbol() for n in neighbors]
                    
                    # Simple heuristics for common cases
                    if len(neighbors) == 1:
                        # Single bonded DU - likely H or O
                        if neighbors[0].GetSymbol() in ['C', 'N']:
                            atom.SetAtomicNum(1)  # H
                            logger.debug(f"Fixed DU atom to H (single bond to {neighbors[0].GetSymbol()})")
                    elif len(neighbors) == 2:
                        # Double bonded DU - likely O or N
                        if all(n.GetSymbol() == 'C' for n in neighbors):
                            atom.SetAtomicNum(8)  # O
                            logger.debug(f"Fixed DU atom to O (bonded to two C)")
                        else:
                            atom.SetAtomicNum(7)  # N
                            logger.debug(f"Fixed DU atom to N (bonded to mixed atoms)")
                
                # Try to sanitize the fixed molecule
                try:
                    Chem.SanitizeMol(mol_copy)
                    return mol_copy.GetMol()
                except:
                    logger.warning(f"Sanitization failed after DU fixing, returning original")
                    return mol
            else:
                # Atom counts match - try direct mapping
                logger.debug(f"Atom counts match, attempting direct template mapping")
                
                # Create new molecule from template with coordinates from broken mol
                new_mol = Chem.RWMol(template_mol)
                
                # Copy coordinates if conformer exists
                if mol.GetNumConformers() > 0:
                    conf = Chem.Conformer(new_mol.GetNumAtoms())
                    old_conf = mol.GetConformer()
                    
                    for i in range(min(mol.GetNumAtoms(), new_mol.GetNumAtoms())):
                        pos = old_conf.GetAtomPosition(i)
                        conf.SetAtomPosition(i, pos)
                    
                    new_mol.RemoveAllConformers()
                    new_mol.AddConformer(conf)
                
                try:
                    Chem.SanitizeMol(new_mol)
                    logger.info(f"Successfully fixed DU atoms using template mapping")
                    return new_mol.GetMol()
                except:
                    logger.warning(f"Template mapping failed, returning original with partial DU fixes")
                    return mol
                    
        except Exception as e:
            logger.error(f"DU atom fixing failed: {e}")
            return mol

    def retrieve_csv(self):
        score_csv: Path = self.plants_dir / "ranking.csv"
        score_df = pd.read_csv(score_csv)

        return score_df

    def main(self):
        """Process docked ligands and merge with scores."""
        mol_name = self.plants_dir.name
        logger.info(f"=== Starting Convert_Plants.main() for {mol_name} ===")
        
        # Load molecules from mol2 file
        logger.info(f"Loading molecules from MOL2 file for {mol_name}")
        rdmol_df = self.get_rdmol_df()
        logger.info(f"Loaded rdmol_df with {len(rdmol_df)} rows for {mol_name}")
        
        # Load scores from CSV
        logger.info(f"Loading scores from CSV for {mol_name}")
        score_df = self.retrieve_csv()
        logger.info(f"Loaded score_df with {len(score_df)} rows for {mol_name}")
        
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
        
        # Process the final molecule names
        if not comb_df.empty:
            try:
                # Clean up names: split the name, filter out 'entry', and take the first parts
                clean_names = []
                for entry in comb_df["LIGAND_ENTRY"]:
                    # Split the name and filter out any parts containing 'entry'
                    parts = [part for part in entry.split('_') if 'entry' not in part.lower()]
                    # Take only the first two parts (ligand name and isomer info)
                    base_parts = parts[:min(2, len(parts))]
                    # Join them back with underscore and add Plants suffix
                    clean_names.append('_'.join(base_parts) + "_Plants")
                
                comb_df["LIGAND_ENTRY"] = clean_names
            except Exception as e:
                logger.warning(f"Error processing names: {str(e)}")
                comb_df["LIGAND_ENTRY"] = [f"Molecule_{i}_Plants" for i in range(len(comb_df))]
            
            # Add pose numbers starting from 1 (like Gnina does)
            comb_df["LIGAND_ENTRY"] = [
                f"{entry}-P{i+1}" for i, entry in enumerate(comb_df["LIGAND_ENTRY"])
            ]
        
        return comb_df