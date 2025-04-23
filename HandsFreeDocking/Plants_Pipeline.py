import os
import shutil
import subprocess
from typing import List, Tuple, Dict, Any, Union
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

from rdkit.Chem import PandasTools
from openbabel import pybel, openbabel

try:
    from openeye import oechem
    from openeye import oeomega
    OPENEYE_AVAILABLE = True
except ImportError:
    OPENEYE_AVAILABLE = False

try:
    from .tools.Protein_Preparation import ProteinPreparation_Chimera
    from .tools.tools import pybel_converter, pybel_flagger
    from .tools.CDPK_Utils import CDPK_Runner, stero_enumerator
    from .tools.OpeneEye_Utils import fix_3dmol, get_chirality_and_stereo, gen_3dmol
except:
    from tools.Protein_Preparation import ProteinPreparation_Chimera
    from tools.tools import pybel_converter, pybel_flagger
    from tools.CDPK_Utils import CDPK_Runner, stero_enumerator
    from tools.OpeneEye_Utils import fix_3dmol, get_chirality_and_stereo, gen_3dmol

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
                toolkit: str = "cdpkit"):
        """
        Initialize the Plants docking pipeline
        
        Args:
            workdir: Working directory for docking
            pdb_ID: Path to the PDB file
            crystal_path: Path to the crystal ligand file
            ligands_sdf: Path to the ligands SDF file
            toolkit: Which toolkit to use for ligand preparation ("cdpkit" or "openeye")
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
        
        # Set the toolkit for ligand preparation
        if toolkit.lower() not in ["cdpkit", "openeye"]:
            raise ValueError(f"Toolkit must be either 'cdpkit' or 'openeye', got {toolkit}")
        
        if toolkit.lower() == "openeye" and not OPENEYE_AVAILABLE:
            logger.warning("OpenEye toolkit not available! Falling back to CDPKit.")
            self.toolkit = "cdpkit"
        else:
            self.toolkit = toolkit.lower()

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
        ligands_mol2_folder = self.workdir / "Ligands_Mol2"
        ligands_mol2_folder.mkdir(exist_ok=True)
        
        if self.toolkit == "cdpkit":
            # For CDPKit workflow:
            # 1. First enumerate stereoisomers using RDKit-based function
            ligands_stereo_path = self.workdir / f"{self.ligands_sdf.stem}_stereo.sdf"
            logger.info(f"Enumerating stereoisomers with RDKit for {self.ligands_sdf}")
            ligands_stereo_path = stero_enumerator(self.ligands_sdf, ligands_stereo_path)
            
            # 2. Then prepare the ligands using CDPK
            ligands_sdf_prepared = self.workdir / "ligands_prepared.sdf"
            logger.info(f"Preparing ligands with CDPKit")
            cdpk_runner = CDPK_Runner()
            cdpk_runner.prepare_ligands(ligands_stereo_path, ligands_sdf_prepared)
            
            # 3. Convert to mol2 and add to list
            logger.info(f"Converting prepared ligands to mol2 format")
            for obmol in pybel.readfile("sdf", str(ligands_sdf_prepared)):
                obmol = pybel_flagger(obmol)
                obmol.write("mol2", str(ligands_mol2_folder / f"{obmol.title}.mol2"), overwrite=True)
                
                self.ligands_mol2.append(ligands_mol2_folder / f"{obmol.title}.mol2")
        else:
            # OpenEye method for ligand preparation
            # Get SMILES from SDF file first to use with gen_3dmol
            logger.info(f"Extracting SMILES from SDF file to prepare with OpenEye toolkit")
            molecules_data = []
            
            ifs = oechem.oemolistream()
            if not ifs.open(str(self.ligands_sdf)):
                raise FileNotFoundError(f"Cannot open {self.ligands_sdf}")
                
            for oemol in ifs.GetOEGraphMols():
                title = oemol.GetTitle()
                smiles = oechem.OECreateSmiString(oemol)
                molecules_data.append((smiles, title))
            ifs.close()
            
            # Process each molecule with gen_3dmol which properly generates 3D coordinates
            logger.info(f"Generating 3D structures with OpenEye toolkit")
            for smiles, title in molecules_data:
                # Use gen_3dmol instead of fix_3dmol to ensure proper 3D generation
                # This returns a list of stereoisomers with 3D coordinates
                oemol_lst = gen_3dmol(smiles, protonate=True, gen3d=True, enum_isomers=True)
                
                logger.info(f"Generated {len(oemol_lst)} stereoisomers for {title}")
                
                # Process each stereoisomer
                for j, enantiomer in enumerate(oemol_lst):
                    # Use 'Iso' naming to be consistent with CDPKit pattern
                    enantiomer_name = f"{title}_Iso{j}"
                    enantiomer.SetTitle(enantiomer_name)
                    
                    # Get and store chirality information
                    chirality_info = get_chirality_and_stereo(enantiomer)
                    if chirality_info:
                        oechem.OESetSDData(enantiomer, "ChiralInfo", chirality_info)
                    
                    # Save to mol2 file
                    outpath = ligands_mol2_folder / f"{enantiomer_name}.mol2"
                    ofs = oechem.oemolostream(str(outpath))
                    oechem.OEWriteMolecule(ofs, enantiomer)
                    ofs.close()
                    
                    # Append the path to the list
                    self.ligands_mol2.append(outpath)

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

                converter = Convert_Plants(docked_plants)
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
    def __init__(self, plants_mol: Path):
        self.plants_mol = plants_mol
        self.plants_dir: Path = self.plants_mol.parent

    def get_rdmol_df(self):
        rd_mols: List = dm.read_mol2file(self.plants_mol)

        rows = []
        for rd_mol in rd_mols:
            row = {
                "LIGAND_ENTRY": rd_mol.GetProp("_Name"),
                "Molecule": rd_mol
            }
            rows.append(row)

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
    
if __name__ == "__main__":
    workdir = Path("./TMP_Docking")
    pdb_ID = Path("./0_Examples/8gcy.pdb")
    crystal_path = Path("./0_Examples/Crystal.sdf")
    ligands_sdf = Path("./0_Examples/some_ligands.sdf")

    # Initialize and run the Plants docking pipeline
    plants_pipeline = Plants_Docking(workdir, pdb_ID, crystal_path, ligands_sdf, toolkit="openeye")
    plants_pipeline.main(n_confs=10, n_cpus=2)