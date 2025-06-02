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
from rdkit import Chem

import pandas as pd
import numpy as np

from rdkit.Chem import PandasTools
from openbabel import pybel, openbabel

from .tools.Protein_Preparation import ProteinPreparation_Chimera
from .tools.tools import pybel_converter, pybel_flagger
from .tools.Ligand_Preparation import LigandPreparator

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
                protonation_method: str = "cdp", tautomer_score_threshold: Optional[float] = None):
        """
        Initialize the RxDock docking pipeline
        
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
        self.ligands_splitted: List[Path] = []
        self.protein_cleaned = None
        self.protein_prepared = None
        self.rxdock_prm_file: Optional[Path] = None
        self.cavity_file: Optional[Path] = None

        # Create required directories
        # rxdock dir / output: for rxdock docking output (consistent with Plants/output)
        self.docked_output = self.workdir / "output"
        self.docked_output.mkdir(exist_ok=True, parents=True)

        self.docked_final_dir: Path = self.workdir / "output"
        self.docked_final_dir.mkdir(exist_ok=True)

        self.docked_rxdock: List[Path] = []
        
        # Set the protonation method for ligand preparation
        if protonation_method.lower() not in ["cdp", "oe", "scrubber"]:
            raise ValueError(f"Protonation method must be 'cdp', 'oe', or 'scrubber', got {protonation_method}")
        
        self.protonation_method = protonation_method.lower()
        self.tautomer_score_threshold = tautomer_score_threshold

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
        
        # Save individual molecules for docking
        logger.info(f"Splitting prepared ligands into individual files")
        for mol in prepared_mols:
            if mol is None:
                continue
                
            mol_name = mol.GetProp("_Name")
            ligand_split = ligands_splitted_path / f"{mol_name}.sdf"
            
            self.ligands_splitted.append(ligand_split.absolute())
            preparator.save_to_sdf([mol], ligand_split)
                    
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
            # Output file base name (without extension) - include software name
            output_base = self.docked_output / f"{ligand_sdf.stem}_Rxdock"
            
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
        
        # Process each docked ligand by updating the original SDF files with proper naming
        all_results = []
        processed_sdf_files = []
        
        for docked_output in self.docked_rxdock:
            try:
                # Process the docked output - update the original file
                converter = Convert_RxDock(docked_output)
                df, updated_file = converter.main()
                
                if not df.empty:
                    all_results.append(df)
                
                # Add the updated file to our list if successful
                if updated_file:
                    processed_sdf_files.append(updated_file)
                    
            except Exception as e:
                logger.error(f"Error processing results for {docked_output}: {str(e)}")
                
        # Combine all results
        combined_df = pd.DataFrame()
        if all_results:
            try:
                combined_df = pd.concat(all_results, ignore_index=True)
                
                logger.info(f"Updated {len(processed_sdf_files)} RxDock output files with proper molecule naming")
                
            except Exception as e:
                logger.error(f"Error combining results: {str(e)}")
                
        return {
            "docked_ligands": self.docked_rxdock,
            "results_df": combined_df,
            "processed_sdf_files": processed_sdf_files  # Return the list of updated original SDF files
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
    def __init__(self, rxdock_output: Path, output_dir: Optional[Path] = None):
        """
        Initialize the Convert_RxDock class for processing RxDock output
        
        Args:
            rxdock_output: Path to RxDock output file (.sd)
            output_dir: Directory to save processed output files (defaults to parent dir of rxdock_output)
        """
        self.rxdock_output = rxdock_output
        self.rxdock_dir: Path = self.rxdock_output.parent
        self.output_dir: Path = output_dir if output_dir else self.rxdock_dir
        self.processed_files: List[Path] = []
        
    def get_rdmol_df(self) -> pd.DataFrame:
        """
        Convert RxDock output to DataFrame with RDKit molecules
        
        Returns:
            DataFrame with RDKit molecules and metadata
        """
        try:
            # Read molecules from SD file with sanitize=False to handle RxDock output issues
            mols = list(dm.read_sdf(self.rxdock_output, as_df=False, sanitize=False))
            
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
            # Read molecules with properties from SD file with sanitize=False
            mols_df = dm.read_sdf(self.rxdock_output, as_df=True, sanitize=False)
            
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
            
            # IMPORTANT: Keep the original SCORE column (don't rename to 'Score') 
            # This ensures it's recognized by Wrapper_Docking._get_docked_dataframe
            # Add a separate Score column for internal use if needed
            main_score_col = [col for col in score_cols if 'SCORE' in col.upper()][0]
            if main_score_col != "SCORE":
                # If the main score column isn't exactly "SCORE", rename it
                score_df["SCORE"] = score_df[main_score_col]
            
            logger.info(f"Found scores for {len(score_df)} poses")
            return score_df
            
        except Exception as e:
            logger.error(f"Error retrieving scores: {str(e)}")
            return pd.DataFrame()
    
    def update_sdf_file(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Update the RxDock output SDF file with proper molecule naming convention.
        Instead of creating multiple files, this updates the original file with proper
        molecule names for each pose within the same file.
        
        Args:
            df: DataFrame with molecules and scores
            
        Returns:
            Path to the updated SDF file or None if failed
        """
        try:
            if df.empty:
                logger.warning("No data to update in SDF file")
                return None
                
            # Extract the base ligand name without the _Rxdock suffix
            base_name = self.rxdock_output.stem
            if base_name.endswith("_Rxdock"):
                # Remove the _Rxdock suffix to get the original ligand name
                ligand_base_name = base_name[:-7]  # Remove "_Rxdock" (7 characters)
            else:
                ligand_base_name = base_name
                
            molecules = []
            
            # Reset the index after sorting to ensure pose numbers match sorted order
            df = df.reset_index(drop=True)
            
            # For each molecule in the dataframe
            for i, row in df.iterrows():
                # Get molecule
                mol = row["Molecule"]
                
                # Set the name property with proper convention: {ligand_base_name}_Rxdock-P{i+1}
                pose_name = f"{ligand_base_name}_Rxdock-P{i+1}"
                mol.SetProp("_Name", pose_name)  # Set the molecule name property
                mol.SetProp("LIGAND_ENTRY", pose_name)  # Also set as a property for dataframe
                
                # Make sure SCORE is set properly
                if "SCORE" in row and not pd.isna(row["SCORE"]):
                    # Keep the original SCORE property (important for _get_docked_dataframe)
                    if not mol.HasProp("SCORE"):
                        mol.SetProp("SCORE", str(row["SCORE"]))
                
                # Add software property
                mol.SetProp("Software", "rxdock")
                
                # Add molecule to the list
                molecules.append(mol)
            
            # Write all molecules back to the original file
            with dm.without_rdkit_log():
                dm.to_sdf(molecules, self.rxdock_output)
            
            logger.info(f"Updated SDF file with {len(molecules)} properly named poses: {self.rxdock_output}")
            return self.rxdock_output
            
        except Exception as e:
            logger.error(f"Error updating SDF file: {str(e)}")
            return None
    
    def main(self) -> Tuple[pd.DataFrame, Optional[Path]]:
        """
        Process docked ligands, merge with scores, and update the original SDF file
        with proper molecule naming conventions for each pose.
        
        Returns:
            Tuple containing:
                - DataFrame with docked poses and scores
                - Path to the updated SDF file or None if failed
        """
        # Get molecules
        rdmol_df = self.get_rdmol_df()
        
        # Get scores
        score_df = self.retrieve_scores()
        
        # Merge molecules and scores
        if rdmol_df.empty or score_df.empty:
            logger.warning("Either molecules or scores are missing")
            return pd.DataFrame(), None
            
        try:
            # Merge on LIGAND_ENTRY
            comb_df = pd.merge(rdmol_df, score_df, on="LIGAND_ENTRY")
            
            # Process names to match standard format
            base_name = self.rxdock_output.stem
            
            # Format entry names: {base_name}_Rxdock-P{i+1}
            comb_df["LIGAND_ENTRY"] = [f"{base_name}_Rxdock-P{i+1}" for i in range(len(comb_df))]
            
            # Add software name for identification (lowercase for consistency with Wrapper_Docking)
            comb_df["Software"] = "rxdock"
            
            # Add protein path for reference
            comb_df["Protein_Path"] = str(self.rxdock_output)
            
            # Sort by score (lower is better for RxDock)
            if "SCORE" in comb_df.columns:
                comb_df = comb_df.sort_values("SCORE", ascending=True)
            
            # Update the original SDF file with proper molecule naming
            updated_file = self.update_sdf_file(comb_df)
            
            # If update was successful, add to processed_files list
            if updated_file:
                self.processed_files = [updated_file]
                logger.info(f"Successfully updated RxDock output file: {updated_file}")
            
            return comb_df, updated_file
            
        except Exception as e:
            logger.error(f"Error processing RxDock output: {str(e)}")
            return pd.DataFrame(), None


if __name__ == "__main__":
    workdir = Path("./TMP_Docking")
    pdb_ID = Path("./0_Examples/8gcy.pdb")
    crystal_path = Path("./0_Examples/Crystal.sdf")
    ligands_sdf = Path("./0_Examples/some_ligands.sdf")

    # Initialize and run the RxDock docking pipeline
    rxdock_pipeline = RxDock_Docking(workdir, pdb_ID, crystal_path, ligands_sdf, protonation_method="cdp", tautomer_score_threshold=2.0)
    rxdock_pipeline.main(n_poses=10, n_cpus=2)
