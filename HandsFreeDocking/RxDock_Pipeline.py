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


class RxDockNamingStrategy:
    """Handle naming conventions for RxDock output files and molecules."""
    
    @staticmethod
    def extract_ligand_base_name(filename: str) -> str:
        """
        Extract base ligand name, removing software suffix if present.
        
        Args:
            filename: Filename or path to extract base name from
            
        Returns:
            Base ligand name without software suffix
        """
        # Remove file extension and get stem
        stem = Path(filename).stem
        
        # Define suffixes to remove (case-insensitive)
        software_suffixes = ['_Rxdock', '_rxdock', '_RXDOCK']
        
        for suffix in software_suffixes:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        
        return stem
    
    @staticmethod
    def create_pose_name(ligand_base: str, pose_num: int) -> str:
        """
        Create pose name following naming convention.
        
        Args:
            ligand_base: Base ligand name (e.g., "LigComplex1_Iso0_Taut0")
            pose_num: Pose number (1-based)
            
        Returns:
            Formatted pose name (e.g., "LigComplex1_Iso0_Taut0_Rxdock-P1")
        """
        return f"{ligand_base}_Rxdock-P{pose_num}"
    
    @staticmethod
    def create_entry_name(rxdock_output: Path, conf_num: int) -> str:
        """
        Create entry name for internal processing.
        
        Args:
            rxdock_output: Path to RxDock output file
            conf_num: Conformation number (1-based)
            
        Returns:
            Entry name for tracking
        """
        return f"{rxdock_output.stem}_conf_{conf_num}"


class RxDockDataParser:
    """Parse RxDock output files to extract molecules and properties."""
    
    def __init__(self, sd_file: Path):
        """
        Initialize parser for a specific SD file.
        
        Args:
            sd_file: Path to RxDock output SD file
        """
        self.sd_file = sd_file
        
    def read_molecules(self) -> List[Chem.Mol]:
        """
        Read molecules from SD file with proper error handling.
        
        Returns:
            List of RDKit molecule objects
        """
        try:
            # Use sanitize=False to handle RxDock output issues
            molecules = list(dm.read_sdf(self.sd_file, as_df=False, sanitize=False))
            return [mol for mol in molecules if mol is not None]
        except Exception as e:
            logger.error(f"Error reading molecules from {self.sd_file}: {e}")
            return []
    
    def read_properties_dataframe(self) -> pd.DataFrame:
        """
        Read molecule properties as DataFrame.
        
        Returns:
            DataFrame with molecule properties and scores
        """
        try:
            # Use sanitize=False to handle RxDock output issues
            df = dm.read_sdf(self.sd_file, as_df=True, sanitize=False)
            return df if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading properties from {self.sd_file}: {e}")
            return pd.DataFrame()
    
    def extract_scores(self) -> pd.DataFrame:
        """
        Extract score columns from properties.
        
        Returns:
            DataFrame with score information
        """
        props_df = self.read_properties_dataframe()
        
        if props_df.empty:
            return pd.DataFrame()
        
        # Find score columns
        score_cols = [col for col in props_df.columns if 'SCORE' in col.upper()]
        
        if not score_cols:
            logger.warning(f"No score columns found in {self.sd_file}")
            return pd.DataFrame()
        
        # Create entry names for tracking
        naming_strategy = RxDockNamingStrategy()
        props_df["LIGAND_ENTRY"] = [
            naming_strategy.create_entry_name(self.sd_file, i+1) 
            for i in range(len(props_df))
        ]
        
        # Select relevant columns
        result_df = props_df[["LIGAND_ENTRY"] + score_cols].copy()
        
        # Ensure main SCORE column exists
        main_score_col = next((col for col in score_cols if col.upper() == 'SCORE'), None)
        if main_score_col and main_score_col != "SCORE":
            result_df["SCORE"] = result_df[main_score_col]
        
        return result_df


class RxDockFileProcessor:
    """Handle file operations for RxDock output processing."""
    
    @staticmethod
    def update_molecule_names(molecules: List[Chem.Mol], 
                            base_name: str,
                            naming_strategy: RxDockNamingStrategy) -> List[Chem.Mol]:
        """
        Update molecule names using naming strategy.
        
        Args:
            molecules: List of RDKit molecules
            base_name: Base ligand name
            naming_strategy: Naming strategy to use
            
        Returns:
            List of molecules with updated names
        """
        updated_molecules = []
        
        for i, mol in enumerate(molecules):
            if mol is None:
                continue
                
            # Create proper pose name
            pose_name = naming_strategy.create_pose_name(base_name, i + 1)
            
            # Update molecule properties but preserve existing ones
            mol.SetProp("_Name", pose_name)
            mol.SetProp("LIGAND_ENTRY", pose_name)
            mol.SetProp("Software", "rxdock")
            
            # Ensure SCORE is preserved if it exists - this is important for wrapper compatibility
            # The wrapper expects molecules to have SCORE property when reading with PandasTools
            
            updated_molecules.append(mol)
        
        return updated_molecules
    
    @staticmethod
    def write_updated_file(molecules: List[Chem.Mol], output_path: Path) -> bool:
        """
        Write molecules back to file.
        
        Args:
            molecules: List of updated molecules
            output_path: Path to write file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with dm.without_rdkit_log():
                dm.to_sdf(molecules, output_path)
            return True
        except Exception as e:
            logger.error(f"Error writing updated file {output_path}: {e}")
            return False

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
    """
    Refactored RxDock output processor using modular components.
    
    This class coordinates the processing of RxDock output files using
    specialized components for data parsing, naming, and file operations.
    """
    
    def __init__(self, rxdock_output: Path, output_dir: Optional[Path] = None):
        """
        Initialize the Convert_RxDock processor.
        
        Args:
            rxdock_output: Path to RxDock output file (.sd)
            output_dir: Directory to save processed output files (defaults to parent dir of rxdock_output)
        """
        self.rxdock_output = rxdock_output
        self.rxdock_dir: Path = self.rxdock_output.parent
        self.output_dir: Path = output_dir if output_dir else self.rxdock_dir
        self.processed_files: List[Path] = []
        
        # Initialize components
        self.data_parser = RxDockDataParser(rxdock_output)
        self.naming_strategy = RxDockNamingStrategy()
        self.file_processor = RxDockFileProcessor()
        
    def get_rdmol_df(self) -> pd.DataFrame:
        """
        Convert RxDock output to DataFrame with RDKit molecules.
        
        Returns:
            DataFrame with RDKit molecules and metadata
        """
        molecules = self.data_parser.read_molecules()
        
        if not molecules:
            logger.warning(f"No molecules found in {self.rxdock_output}")
            return pd.DataFrame()
            
        # Create DataFrame with molecules using proper naming
        df = pd.DataFrame({
            "LIGAND_ENTRY": [
                self.naming_strategy.create_entry_name(self.rxdock_output, i+1) 
                for i in range(len(molecules))
            ],
            "Molecule": molecules
        })
            
        logger.info(f"Found {len(df)} docked poses in {self.rxdock_output}")
        return df
            
    def retrieve_scores(self) -> pd.DataFrame:
        """
        Extract scores from RxDock output using data parser.
        
        Returns:
            DataFrame with docking scores
        """
        return self.data_parser.extract_scores()
    
    def update_sdf_file(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Update the RxDock output SDF file with proper molecule naming.
        
        Args:
            df: DataFrame with molecules and scores
            
        Returns:
            Path to the updated SDF file or None if failed
        """
        if df.empty:
            logger.warning("No data to update in SDF file")
            return None
            
        try:
            # Extract base ligand name using robust naming strategy
            ligand_base_name = self.naming_strategy.extract_ligand_base_name(
                self.rxdock_output.name
            )
            
            # Get molecules from DataFrame
            molecules = df["Molecule"].tolist()
            
            # Sort DataFrame by score (lower is better for RxDock) and reset index
            if "SCORE" in df.columns:
                df = df.sort_values("SCORE", ascending=True).reset_index(drop=True)
                
            # Update molecule names using file processor
            updated_molecules = self.file_processor.update_molecule_names(
                molecules, ligand_base_name, self.naming_strategy
            )
            
            # Preserve scores in molecule properties
            for mol, (_, row) in zip(updated_molecules, df.iterrows()):
                if "SCORE" in row and not pd.isna(row["SCORE"]):
                    mol.SetProp("SCORE", str(row["SCORE"]))
            
            # Write updated file
            success = self.file_processor.write_updated_file(updated_molecules, self.rxdock_output)
            
            if success:
                logger.info(f"Updated SDF file with {len(updated_molecules)} properly named poses: {self.rxdock_output}")
                return self.rxdock_output
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error updating SDF file: {str(e)}")
            return None
    
    def main(self) -> Tuple[pd.DataFrame, Optional[Path]]:
        """
        Process docked ligands with proper naming conventions.
        
        Returns:
            Tuple containing:
                - DataFrame with docked poses and scores
                - Path to the updated SDF file or None if failed
        """
        # Get molecules and scores using new components
        rdmol_df = self.get_rdmol_df()
        score_df = self.retrieve_scores()
        
        if rdmol_df.empty or score_df.empty:
            logger.warning("Either molecules or scores are missing")
            return pd.DataFrame(), None
            
        try:
            # Merge on LIGAND_ENTRY
            comb_df = pd.merge(rdmol_df, score_df, on="LIGAND_ENTRY")
            
            # Sort by score (lower is better for RxDock)
            if "SCORE" in comb_df.columns:
                comb_df = comb_df.sort_values("SCORE", ascending=True)
            
            # Update entry names to final format for consistency
            ligand_base_name = self.naming_strategy.extract_ligand_base_name(
                self.rxdock_output.name
            )
            comb_df["LIGAND_ENTRY"] = [
                self.naming_strategy.create_pose_name(ligand_base_name, i+1) 
                for i in range(len(comb_df))
            ]
            
            # Add metadata
            comb_df["Software"] = "rxdock"
            comb_df["Protein_Path"] = str(self.rxdock_output)
            
            # Update the original SDF file with proper molecule naming
            updated_file = self.update_sdf_file(comb_df)
            
            # Track processed files
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
