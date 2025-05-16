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
except ImportError:
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

class RxDock_Docking:
    def __init__(self, workdir: Path, pdb_ID: Path, crystal_path: Path, ligands_sdf: Path, 
                toolkit: str = "cdpkit"):
        """
        Initialize the RxDock docking pipeline
        
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
        self.ligands_splitted: List[Path] = []
        self.protein_cleaned = None
        self.protein_prepared = None
        self.rxdock_prm_file: Optional[Path] = None
        self.cavity_file: Optional[Path] = None

        self.docked_output_rxdock: Path = workdir / "output_rxdock"
        self.docked_output_rxdock.mkdir(exist_ok=True)

        self.docked_final_dir: Path = workdir / "output"
        self.docked_final_dir.mkdir(exist_ok=True)

        self.docked_rxdock: List[Path] = []
        
        # Set the toolkit for ligand preparation
        if toolkit.lower() not in ["cdpkit", "openeye"]:
            raise ValueError(f"Toolkit must be either 'cdpkit' or 'openeye', got {toolkit}")
        
        if toolkit.lower() == "openeye" and not OPENEYE_AVAILABLE:
            logger.warning("OpenEye toolkit not available! Falling back to CDPKit.")
            self.toolkit = "cdpkit"
        else:
            self.toolkit = toolkit.lower()

        self._rxdock_env_variable()

    def _rxdock_env_variable(self) -> None:
        """
        Set up RxDock environment variables
        """
        # Set RxDock environment variables
        os.environ['RBT_ROOT'] = '/home/hitesit/Software/rxdock/rxdock_installation'
        os.environ['PATH'] = f"{os.environ['RBT_ROOT']}/bin:{os.environ.get('PATH', '')}"
        os.environ['LD_LIBRARY_PATH'] = f"{os.environ['RBT_ROOT']}/lib/x86_64-linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}"
        # Set RBT_HOME to the absolute path of workdir
        os.environ['RBT_HOME'] = str(self.workdir.resolve())
        
    def _resolve_path(self, path: Path) -> str:
        """
        Resolve a path to its absolute form for use in parameter files
        
        Args:
            path: Path to resolve
            
        Returns:
            Absolute path as a string
        """
        # Convert to absolute path
        abs_path = path.resolve()
        
        # Check if path exists
        if not abs_path.exists():
            logger.warning(f"Path does not exist: {abs_path}")
            
        return str(abs_path)

    def _source_macro(self) -> None:
        """
        Extract protein from PDB file and clean it
        """
        # Grab the protein from the PDB
        protein_path = self.pdb_ID
        reader = pdb.PDBFile.read(protein_path)
        struct_array = reader.get_structure(model=1)

        # Remove all and not protein
        macro_array = struct_array[struc.filter_amino_acids(struct_array)]

        protein_cleaned: Path = self.workdir / f"{self.pdb_ID.stem}_clean.pdb"
        self.protein_cleaned = protein_cleaned

        strucio.save_structure(str(protein_cleaned), macro_array)

    def _prepare_protein(self) -> None:
        """
        Prepare the protein using Chimera
        """
        pdb_mol2 = self.workdir / f"{self.pdb_ID.stem}_prep.mol2"
        chimera_prep = ProteinPreparation_Chimera()
        chimera_prep(self.protein_cleaned, pdb_mol2)

        self.protein_prepared = pdb_mol2

    def _define_binding_site(self) -> None:
        """
        Define the binding site for RxDock using rbcavity
        based on the crystal ligand position
        
        This method creates the main parameter file used for all docking runs and defines
        the binding site using the reference ligand method. It creates two key files:
        1. rxdock.prm - The main parameter file containing receptor and cavity definition
        2. rxdock.as - The cavity file generated by rbcavity
        
        Both files will be used for all subsequent docking runs with different ligands.
        """
        logger.info("Creating RxDock parameter file and defining binding site...")
        
        # Resolve paths to absolute paths
        receptor_path = self._resolve_path(self.protein_prepared)
        crystal_path = self._resolve_path(self.crystal_path)
        
        # Create parameter file for rxdock
        rxdock_prm_template = f"""RBT_PARAMETER_FILE_V1.00
TITLE {self.pdb_ID.stem}_rxdock
RECEPTOR_FILE {receptor_path}
RECEPTOR_FLEX 3.0

##############################################
## CAVITY DEFINITION: REFERENCE LIGAND METHOD
##############################################
SECTION MAPPER
    SITE_MAPPER RbtLigandSiteMapper
    REF_MOL {crystal_path}
    RADIUS 6.0
    SMALL_SPHERE 1.0
    MIN_VOLUME 100
    MAX_CAVITIES 1
    VOL_INCR 0.0
END_SECTION

############################
## CAVITY RESTRAINT PENALTY
############################
SECTION CAVITY
    SCORING_FUNCTION RbtCavityGridSF
    WEIGHT 1.0
END_SECTION
        """
        
        # Write the parameter file
        self.rxdock_prm_file = self.workdir / "rxdock.prm"
        with open(self.rxdock_prm_file, "w") as f:
            f.write(rxdock_prm_template)
            
        # Run rbcavity to define the binding site
        cmd = [
            "rbcavity",
            "-W",  # Write cavity as grid file
            "-d",  # Write cavity description
            "-r",  # Parameter file
            str(self.rxdock_prm_file)
        ]
        
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(result.stdout)
            
            # The cavity file is created with the same name as the parameter file but with .as extension
            cavity_file = self.rxdock_prm_file.with_suffix('.as')
            if cavity_file.exists():
                self.cavity_file = cavity_file
                logger.info(f"Cavity file created: {self.cavity_file}")
            else:
                raise FileNotFoundError(f"Cavity file not created: {cavity_file}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running rbcavity: {e}")
            logger.error(f"Stderr: {e.stderr}")
            raise e

    def prepare_ligands(self) -> None:
        """
        Prepare ligands for docking by generating 3D conformers and enumerating stereoisomers
        
        This method processes the input ligands SDF file to generate properly prepared
        ligands for docking. The process includes:
        1. Enumeration of stereoisomers (if unspecified in the input)
        2. 3D conformation generation
        3. Protonation at physiological pH
        4. Saving individual ligands as separate files for docking
        
        Two toolkits are supported:
        - CDPKit: Open-source cheminformatics toolkit
        - OpenEye: Commercial toolkit with advanced 3D conformer generation (requires license)
        
        The toolkit used is determined by the self.toolkit attribute set during initialization.
        All prepared ligands are stored in the 'ligands_split' directory, and their paths
        are saved in self.ligands_splitted for use in docking.
        
        Returns:
            None
        
        Raises:
            FileNotFoundError: If the input ligands SDF file cannot be opened
        """
        # Create directory for individual ligand files
        ligands_splitted_path: Path = self.workdir / "ligands_split"
        ligands_splitted_path.mkdir(exist_ok=True)
        
        if self.toolkit == "cdpkit":
            # CDPKit workflow (open source option):
            # 1. First enumerate stereoisomers using RDKit-based function
            ligands_stereo_path = self.workdir / f"{self.ligands_sdf.stem}_stereo.sdf"
            logger.info(f"Enumerating stereoisomers with RDKit for {self.ligands_sdf}")
            ligands_stereo_path = stero_enumerator(self.ligands_sdf, ligands_stereo_path)
            
            # 2. Then prepare the ligands using CDPK (add hydrogens, generate 3D coordinates)
            ligand_prepared_path = self.workdir / "ligands_prepared.sdf"
            logger.info(f"Preparing ligands with CDPKit")
            cdpk_runner = CDPK_Runner()
            cdpk_runner.prepare_ligands(ligands_stereo_path, ligand_prepared_path)
            
            # 3. Split into individual files (one ligand per file for docking)
            logger.info(f"Splitting prepared ligands into individual files")
            for mol in Chem.SDMolSupplier(str(ligand_prepared_path)):
                if mol is None:
                    continue
                    
                mol_name = mol.GetProp("_Name")
                ligand_split = ligands_splitted_path / f"{mol_name}.sdf"
                
                self.ligands_splitted.append(ligand_split.absolute())
                Chem.SDWriter(str(ligand_split)).write(mol)
        else:
            # OpenEye method for ligand preparation (commercial option with advanced features)
            # Get SMILES from SDF file first to use with gen_3dmol
            logger.info(f"Extracting SMILES from SDF file to prepare with OpenEye toolkit")
            molecules_data = []
            
            # Open the SDF file with OpenEye tools
            ifs = oechem.oemolistream()
            if not ifs.open(str(self.ligands_sdf)):
                raise FileNotFoundError(f"Unable to open {self.ligands_sdf}")
                
            # Extract molecule titles and SMILES representations
            for oemol in ifs.GetOEGraphMols():
                title = oemol.GetTitle()
                smiles = oechem.OECreateSmiString(oemol)
                molecules_data.append((smiles, title))
            ifs.close()
            
            # Process each molecule with gen_3dmol to get proper 3D coordinates and stereoisomers
            logger.info(f"Generating 3D structures with OpenEye toolkit")
            for smiles, title in molecules_data:
                # gen_3dmol returns a list of stereoisomers with 3D coordinates
                # protonate=True adds hydrogens, gen3d=True generates 3D coords, enum_isomers=True enumerates stereoisomers
                oemol_lst = gen_3dmol(smiles, protonate=True, gen3d=True, enum_isomers=True)
                
                logger.info(f"Generated {len(oemol_lst)} stereoisomers for {title}")
                
                # Process each stereoisomer
                for j, enantiomer in enumerate(oemol_lst):
                    # Use 'Iso' naming convention to be consistent with CDPKit pattern across pipelines
                    enantiomer_name = f"{title}_Iso{j}"
                    enantiomer.SetTitle(enantiomer_name)
                    
                    # Get and store chirality information for reference
                    chirality_info = get_chirality_and_stereo(enantiomer)
                    if chirality_info:
                        oechem.OESetSDData(enantiomer, "ChiralInfo", chirality_info)
                    
                    # Save to individual SDF file
                    ligand_split = ligands_splitted_path / f"{enantiomer_name}.sdf"
                    self.ligands_splitted.append(ligand_split.absolute())
                    
                    ofs = oechem.oemolostream(str(ligand_split))
                    oechem.OEWriteMolecule(ofs, enantiomer)
                    ofs.close()
                    
        logger.info(f"Successfully prepared {len(self.ligands_splitted)} ligands for docking")

    # The write_prm method has been removed as RxDock uses a single parameter file for all ligands,
    # which is created in the _define_binding_site method (rxdock.prm)

    def _save_to_sdf(self, df: pd.DataFrame, name: str) -> Path:
        """
        Save docking results to SDF file
        
        Args:
            df: DataFrame with docking results containing RDKit molecules
            name: Name for the output file
            
        Returns:
            Path to the saved SDF file
        """
        output_file = self.docked_final_dir / f"{name}.sdf"
        
        if not "ROMol" in df.columns:
            logger.warning("No RDKit molecules found in DataFrame. Creating empty SDF file.")
            with open(output_file, "w") as f:
                f.write("")
            return output_file
        
        # Ensure molecule column is named 'ROMol' for PandasTools
        if "Molecule" in df.columns and not "ROMol" in df.columns:
            df["ROMol"] = df["Molecule"]
            
        # Save molecules to SDF
        PandasTools.WriteSDF(df, str(output_file), molColName="ROMol", properties=list(df.columns))
        
        logger.info(f"Saved docking results to {output_file}")
        return output_file
        
    def main(self, n_poses: int = 50, n_cpus: int = 1) -> Dict[str, Any]:
        """
        Run the RxDock docking pipeline
        
        Args:
            n_poses: Number of poses to generate per ligand
            n_cpus: Number of CPU cores to use for parallel processing
            
        Returns:
            Dictionary with docking results
        """
        logger.info("Starting RxDock docking pipeline...")
        
        # Step 1: Source macro (clean protein)
        logger.info("Step 1: Sourcing macro (cleaning protein)...")
        self._source_macro()
        
        # Step 2: Prepare protein
        logger.info("Step 2: Preparing protein...")
        self._prepare_protein()
        
        # Step 3: Define binding site
        logger.info("Step 3: Defining binding site...")
        self._define_binding_site()
        
        # Step 4: Prepare ligands (generate 3D conformers, enumerate stereoisomers)
        logger.info("Step 4: Preparing ligands...")
        self.prepare_ligands()
        
        # Step 5: Run docking with each ligand
        logger.info("Step 5: Running docking...")
        
        # Check if we have ligands to dock
        if not self.ligands_splitted:
            raise ValueError("No ligands available for docking")
            
        # Verify that we have the parameter file and cavity file from _define_binding_site
        if not self.rxdock_prm_file or not self.cavity_file:
            raise ValueError("RxDock parameter file or cavity file not defined. Run _define_binding_site first.")
            
        # Create configuration for each ligand (using the same parameter file for all)
        docking_configs = []
        for ligand_sdf in self.ligands_splitted:
            # Output file base name (without extension)
            output_base = self.docked_output_rxdock / ligand_sdf.stem
            
            # Add to list of configurations - all ligands use the same parameter file
            docking_configs.append((ligand_sdf, self.rxdock_prm_file, output_base))
            
        # Run docking in parallel
        if n_cpus > 1 and len(docking_configs) > 1:
            logger.info(f"Running docking in parallel with {n_cpus} CPUs...")
            
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=n_cpus) as executor:
                futures = []
                
                for ligand_sdf, prm_file, output_base in docking_configs:
                    future = executor.submit(
                        self.runner, 
                        ligand_sdf=ligand_sdf, 
                        prm_file=prm_file, 
                        output_base=output_base,
                        n_poses=n_poses
                    )
                    futures.append(future)
                    
                # Track progress
                for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Docking progress")):
                    try:
                        docked_output = future.result()
                        if docked_output:
                            self.docked_rxdock.append(docked_output)
                    except Exception as e:
                        logger.error(f"Error in docking job {i}: {str(e)}")
        else:
            logger.info("Running docking sequentially...")
            
            # Run docking sequentially
            for ligand_sdf, prm_file, output_base in tqdm(docking_configs, desc="Docking progress"):
                try:
                    docked_output = self.runner(
                        ligand_sdf=ligand_sdf, 
                        prm_file=prm_file, 
                        output_base=output_base,
                        n_poses=n_poses
                    )
                    if docked_output:
                        self.docked_rxdock.append(docked_output)
                except Exception as e:
                    logger.error(f"Error docking {ligand_sdf.name}: {str(e)}")
                    
        logger.info(f"Completed docking for {len(self.docked_rxdock)} ligands")
        
        # Step 6: Process results
        logger.info("Step 6: Processing docking results...")
        
        # Process each docked ligand
        all_results = []
        for docked_output in self.docked_rxdock:
            try:
                # Process the docked output
                converter = Convert_RxDock(docked_output)
                df = converter.main()
                
                if not df.empty:
                    all_results.append(df)
            except Exception as e:
                logger.error(f"Error processing results for {docked_output}: {str(e)}")
                
        # Combine all results
        combined_df = pd.DataFrame()
        if all_results:
            try:
                combined_df = pd.concat(all_results, ignore_index=True)
                
                # Save to SDF
                self._save_to_sdf(combined_df, "rxdock_results")
            except Exception as e:
                logger.error(f"Error combining results: {str(e)}")
                
        return {
            "docked_ligands": self.docked_rxdock,
            "results_df": combined_df
        }
    
    def runner(self, ligand_sdf: Path, prm_file: Path, output_base: Path, n_poses: int = 50) -> Optional[Path]:
        """
        Run RxDock docking for a single ligand
        
        This method executes rbdock for a single ligand using:
        - The ligand SDF file
        - The parameter file created in _define_binding_site
        - The standard dock.prm file (which is part of RxDock installation)
        
        Args:
            ligand_sdf: Path to ligand SDF file
            prm_file: Path to the RxDock parameter file (rxdock.prm)
            output_base: Base path for output files
            n_poses: Number of poses to generate
            
        Returns:
            Path to docking output file (.sd) or None if failed
        """
        try:
            # Resolve paths to absolute paths
            ligand_path = self._resolve_path(ligand_sdf)
            param_path = self._resolve_path(prm_file)
            
            # For output base, make sure the directory exists
            output_dir = output_base.parent
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = str(output_base.resolve())
            
            # Output file for docking results
            output_file = f"{output_path}.sd"
            
            # Command to run RxDock
            # Note: dock.prm is a standard file that comes with RxDock installation
            cmd = [
                "rbdock",
                "-i", ligand_path,     # Input ligand (absolute path)
                "-o", output_path,     # Output prefix (absolute path)
                "-r", param_path,      # Parameter file (rxdock.prm with absolute path)
                "-p", "dock.prm",      # Standard docking protocol file (in RxDock path)
                "-n", str(n_poses)     # Number of poses
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check if output file was created
            output_path = Path(f"{output_base}.sd")
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Docking completed: {output_path}")
                return output_path
            else:
                logger.warning(f"Docking completed but output file is empty or missing: {output_path}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running rbdock: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error in docking runner: {str(e)}")
            return None


class Convert_RxDock:
    def __init__(self, rxdock_output: Path):
        """
        Initialize the Convert_RxDock class for processing RxDock output
        
        Args:
            rxdock_output: Path to RxDock output file (.sd)
        """
        self.rxdock_output = rxdock_output
        self.rxdock_dir: Path = self.rxdock_output.parent
        
    def get_rdmol_df(self) -> pd.DataFrame:
        """
        Convert RxDock output to DataFrame with RDKit molecules
        
        Returns:
            DataFrame with RDKit molecules and metadata
        """
        try:
            # Read molecules from SD file
            mols = list(dm.read_sdf(self.rxdock_output, as_df=False))
            
            if not mols:
                logger.warning(f"No molecules found in {self.rxdock_output}")
                return pd.DataFrame()
                
            # Create DataFrame with molecules
            df = pd.DataFrame({
                "LIGAND_ENTRY": [f"{self.rxdock_output.stem}_conf_{i+1}" for i in range(len(mols))],
                "Molecule": mols
            })
                
            logger.info(f"Found {len(df)} docked poses in {self.rxdock_output}")
            return df
                
        except Exception as e:
            logger.error(f"Error reading RxDock output: {str(e)}")
            return pd.DataFrame()
            
    def retrieve_scores(self) -> pd.DataFrame:
        """
        Extract scores from RxDock output
        
        Returns:
            DataFrame with docking scores
        """
        try:
            # Read molecules with properties from SD file
            mols_df = dm.read_sdf(self.rxdock_output, as_df=True)
            
            if mols_df.empty:
                logger.warning(f"No molecules found in {self.rxdock_output}")
                return pd.DataFrame()
                
            # Get column names that contain score information
            score_cols = [col for col in mols_df.columns if 'SCORE' in col.upper()]
            
            if not score_cols:
                logger.warning(f"No score columns found in {self.rxdock_output}")
                return pd.DataFrame()
                
            # Create entry names that match the get_rdmol_df method
            mols_df["LIGAND_ENTRY"] = [f"{self.rxdock_output.stem}_conf_{i+1}" for i in range(len(mols_df))]    

            # Select only the score columns and LIGAND_ENTRY
            score_df = mols_df[["LIGAND_ENTRY"] + score_cols]
            
            # Rename the primary score column to 'Score' for consistency
            main_score_col = [col for col in score_cols if 'SCORE' in col.upper()][0]
            score_df = score_df.rename(columns={main_score_col: "Score"})
            
            logger.info(f"Found scores for {len(score_df)} poses")
            return score_df
            
        except Exception as e:
            logger.error(f"Error retrieving scores: {str(e)}")
            return pd.DataFrame()
    
    def main(self) -> pd.DataFrame:
        """
        Process docked ligands and merge with scores
        
        Returns:
            DataFrame with docked poses and scores
        """
        # Get molecules
        rdmol_df = self.get_rdmol_df()
        
        # Get scores
        score_df = self.retrieve_scores()
        
        # Merge molecules and scores
        if rdmol_df.empty or score_df.empty:
            logger.warning("Either molecules or scores are missing")
            return pd.DataFrame()
            
        try:
            # Merge on LIGAND_ENTRY
            comb_df = pd.merge(rdmol_df, score_df, on="LIGAND_ENTRY")
            
            # Process names to match standard format
            base_name = self.rxdock_output.stem
            
            # Format entry names: {base_name}_RxDock-P{pose_number}
            comb_df["LIGAND_ENTRY"] = [f"{base_name}_RxDock-P{i+1}" for i in range(len(comb_df))]
            
            # Add software name for identification
            comb_df["Software"] = "RxDock"
            
            # Add protein path for reference
            comb_df["Protein_Path"] = str(self.rxdock_output)
            
            # Sort by score (lower is better for RxDock)
            comb_df = comb_df.sort_values("Score", ascending=True)
            
            return comb_df
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return pd.DataFrame()


if __name__ == "__main__":
    workdir = Path("./TMP_Docking")
    pdb_ID = Path("./0_Examples/8gcy.pdb")
    crystal_path = Path("./0_Examples/Crystal.sdf")
    ligands_sdf = Path("./0_Examples/some_ligands.sdf")

    # Initialize and run the RxDock docking pipeline
    rxdock_pipeline = RxDock_Docking(workdir, pdb_ID, crystal_path, ligands_sdf, toolkit="cdpkit")
    rxdock_pipeline.main(n_poses=10, n_cpus=2)
