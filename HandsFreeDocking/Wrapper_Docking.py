import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import tempfile
import sys

# # Add the current directory to the path so we can import src
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import datamol as dm
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.preprocessing import MinMaxScaler

# Check if OpenEye is available
try:
    from openeye import oechem
    from openeye import oeomega
    OPENEYE_AVAILABLE = True
except ImportError:
    OPENEYE_AVAILABLE = False

# Import the docking pipelines
from HandsFreeDocking.Plants_Pipeline import Plants_Docking
from HandsFreeDocking.Gnina_Pipeline import Gnina_Docking
from HandsFreeDocking.RxDock_Pipeline import RxDock_Docking

# Try to import OpenEye pipeline, but don't fail if it's not available
try:
    from HandsFreeDocking.OpenEye_Pipeline import OpenEye_Docking
    OPENEYE_PIPELINE_AVAILABLE = True
except ImportError:
    OPENEYE_PIPELINE_AVAILABLE = False

# Import utilities if available
try:
    from HandsFreeDocking.tools.OpeneEye_Utils import gen_3dmol
except ImportError:
    # Define a placeholder function if OpenEye utils are not available
    def gen_3dmol(*args, **kwargs):
        raise ImportError("OpenEye utilities not available")

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineDocking:
    """
    Wrapper class for running multiple docking software on the same set of ligands and protein.
    
    This class provides a unified interface to run Plants, Gnina, and OpenEye docking
    with flexible input formats (SDF or SMILES) and collects results in separate dataframes.
    """
    
    def __init__(self, 
                workdir: Path, 
                docking_software: List[str], 
                settings: Tuple[int, int], 
                protein_pdb: Path, 
                ligands_input: Path, 
                crystal_sdf: Path,
                toolkit: str = "cdpkit"):
        """
        Initialize the PipelineDocking wrapper.
        
        Args:
            workdir: Working directory for docking
            docking_software: List of docking software to use ("plants", "gnina", "openeye")
            settings: Tuple of (n_conformers, n_cpus)
            protein_pdb: Path to the protein PDB file
            ligands_input: Path to the ligands file (SDF or SMILES)
            crystal_sdf: Path to the crystal ligand SDF file
            toolkit: Toolkit to use for ligand preparation ("cdpkit" or "openeye")
        """
        # Validate inputs
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True, parents=True)
        
        self.docking_software = [s.lower() for s in docking_software]
        for software in self.docking_software:
            if software not in ["plants", "gnina", "openeye", "rxdock"]:
                raise ValueError(f"Unsupported docking software: {software}")
            
            if software == "openeye" and not OPENEYE_PIPELINE_AVAILABLE:
                raise ImportError("OpenEye docking pipeline not available")
        
        # Store settings and file paths
        self.n_confs, self.n_cpus = settings
        self.protein_pdb = protein_pdb
        self.ligands_input = ligands_input
        self.crystal_sdf = crystal_sdf.absolute()
        
        # Validate toolkit
        if toolkit.lower() not in ["cdpkit", "openeye"]:
            raise ValueError(f"Toolkit must be either 'cdpkit' or 'openeye', got {toolkit}")
            
        if toolkit.lower() == "openeye" and not OPENEYE_AVAILABLE:
            logger.warning("OpenEye toolkit not available! Falling back to CDPKit.")
            self.toolkit = "cdpkit"
        else:
            self.toolkit = toolkit.lower()
        
        # Detect input format
        self.input_format = self._detect_input_format()
        
        # Placeholder for results
        self.results = {}
    
    def _detect_input_format(self) -> str:
        """Detect the format of ligands_input file."""
        suffix = self.ligands_input.suffix.lower()
        
        if suffix in ['.sdf', '.mol']:
            return "sdf"
        elif suffix in ['.smi', '.smiles', '.csv', '.txt']:
            # Check if it's a CSV/TXT with SMILES
            try:
                df = pd.read_csv(self.ligands_input)
                if 'SMILES' in df.columns:
                    return "smiles"
            except:
                # Try to read as SMILES file
                with open(self.ligands_input, 'r') as f:
                    line = f.readline().strip()
                    # Simple check: if first line contains a valid SMILES
                    if ' ' in line and not line.startswith('#'):
                        return "smiles"
        
        raise ValueError(f"Unrecognized input format for {self.ligands_input}. "
                         f"Supported formats: SDF or SMILES file.")
    
    def _process_input_to_sdf(self) -> Path:
        """Convert input to SDF if needed."""
        if self.input_format == "sdf":
            logger.info(f"Using SDF input directly: {self.ligands_input}")
            return self.ligands_input
        
        # For SMILES input, convert to SDF
        logger.info(f"Converting SMILES to SDF: {self.ligands_input}")
        ligands_sdf = self.workdir / "processed_ligands.sdf"
        
        if self.toolkit == "cdpkit":
            # Read SMILES file
            smiles_data = []
            
            try:
                df = pd.read_csv(self.ligands_input)
                if 'SMILES' in df.columns:
                    # If ID column exists, use it, otherwise create sequential IDs
                    id_col = 'ID' if 'ID' in df.columns else None
                    
                    for i, row in df.iterrows():
                        smiles = row['SMILES']
                        mol_id = row[id_col] if id_col else f"Mol_{i}"
                        smiles_data.append((smiles, mol_id))
            except:
                # Try reading as SMILES file
                with open(self.ligands_input, 'r') as f:
                    for i, line in enumerate(f):
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                smiles, mol_id = parts[0], parts[1]
                            else:
                                smiles, mol_id = parts[0], f"Mol_{i}"
                            smiles_data.append((smiles, mol_id))
            
            # Convert SMILES to 3D molecules and write to SDF
            writer = Chem.SDWriter(str(ligands_sdf))
            for smiles, mol_id in smiles_data:
                mol = dm.to_mol(smiles)
                if mol:
                    mol = dm.conformers.generate(mol)
                    mol.SetProp("_Name", mol_id)
                    writer.write(mol)
                else:
                    logger.warning(f"Failed to process SMILES: {smiles} ({mol_id})")
            writer.close()
            
        else:  # OpenEye toolkit
            if not OPENEYE_AVAILABLE:
                raise ImportError("OpenEye toolkit not available for SMILES conversion")
                
            # Similar approach as with the Fake_Wrapper.py
            ofs = oechem.oemolostream(str(ligands_sdf))
            
            try:
                df = pd.read_csv(self.ligands_input)
                if 'SMILES' in df.columns:
                    for i, row in df.iterrows():
                        smile = row['SMILES']
                        mol_id = row['ID'] if 'ID' in df.columns else f"Mol_{i}"
                        
                        # Use OpenEye to generate 3D
                        oemol_lst = gen_3dmol(smile, protonate=True, gen3d=True, enum_isomers=True)
                        
                        for j, oemol in enumerate(oemol_lst):
                            if len(oemol_lst) > 1:
                                name = f"{mol_id}_Stereo_{j}"
                            else:
                                name = mol_id
                            oemol.SetTitle(name)
                            oechem.OEWriteMolecule(ofs, oemol)
            except:
                # Try reading as SMILES file
                with open(self.ligands_input, 'r') as f:
                    for i, line in enumerate(f):
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                smile, mol_id = parts[0], parts[1]
                            else:
                                smile, mol_id = parts[0], f"Mol_{i}"
                            
                            # Use OpenEye to generate 3D
                            oemol_lst = gen_3dmol(smile, protonate=True, gen3d=True, enum_isomers=True)
                            
                            for j, oemol in enumerate(oemol_lst):
                                if len(oemol_lst) > 1:
                                    name = f"{mol_id}_Stereo_{j}"
                                else:
                                    name = mol_id
                                oemol.SetTitle(name)
                                oechem.OEWriteMolecule(ofs, oemol)
            
            ofs.close()
            
        logger.info(f"Successfully processed input to SDF: {ligands_sdf}")
        return ligands_sdf
    
    def _run_plants_docking(self, ligands_sdf):
        """Run Plants docking."""
        logger.info("Starting Plants docking")
        
        # Create directory for Plants
        plants_dir = self.workdir / "Plants"
        plants_dir.mkdir(exist_ok=True)
        
        # Initialize and run Plants docking
        plants_pipeline = Plants_Docking(
            workdir=plants_dir,
            pdb_ID=self.protein_pdb,
            crystal_path=self.crystal_sdf,
            ligands_sdf=ligands_sdf,
            toolkit=self.toolkit
        )
        
        plants_pipeline.main(n_confs=self.n_confs, n_cpus=self.n_cpus)
        
        # Collect results
        output_dir = plants_dir / "output"
        docked_ligands = list(output_dir.glob("*_Plants.sdf"))
        
        results_df = self._get_docked_dataframe(docked_ligands, "plants")
        logger.info(f"Plants docking completed with {len(docked_ligands)} results")
        
        return results_df
    
    def _run_gnina_docking(self, ligands_sdf):
        """Run Gnina docking."""
        logger.info("Starting Gnina docking")
        
        # Create directory for Gnina
        gnina_dir = self.workdir / "Gnina"
        gnina_dir.mkdir(exist_ok=True)
        
        # Initialize and run Gnina docking
        gnina_pipeline = Gnina_Docking(
            workdir=gnina_dir,
            pdb_ID=self.protein_pdb,
            crystal_path=self.crystal_sdf,
            ligands_sdf=ligands_sdf,
            toolkit=self.toolkit
        )
        
        gnina_pipeline.non_covalent_run(n_confs=self.n_confs, n_cpus=self.n_cpus)
        
        # Collect results
        output_dir = gnina_dir / "output"
        docked_ligands = list(output_dir.glob("*_Gnina.sdf"))
        
        # Get Gnina results
        results_df = self._get_docked_dataframe(docked_ligands, "gnina")
        logger.info(f"Gnina docking completed with {len(docked_ligands)} results")
        
        # Also create a copy for Smina using minimizedAffinity instead of CNNaffinity
        if not results_df.empty:
            smina_df = self._get_docked_dataframe(docked_ligands, "smina")
            logger.info(f"Smina results extracted from Gnina output")
            return {"gnina": results_df, "smina": smina_df}
        
        return {"gnina": results_df}
    
    def _run_openeye_docking(self, ligands_sdf):
        """Run OpenEye docking."""
        if not OPENEYE_PIPELINE_AVAILABLE:
            raise ImportError("OpenEye docking pipeline not available")
            
        logger.info("Starting OpenEye docking")
        
        # Create directory for OpenEye
        openeye_dir = self.workdir / "OpenEye"
        openeye_dir.mkdir(exist_ok=True)
        
        # For OpenEye we need to prepare a list of tuple (SMILES, ID)
        docking_tuple = []
        
        for mol in Chem.SDMolSupplier(str(ligands_sdf)):
            if mol:
                mol_id = mol.GetProp("_Name")
                smiles = Chem.MolToSmiles(mol)
                docking_tuple.append((smiles, mol_id))
        
        # Initialize and run OpenEye docking
        openeye_pipeline = OpenEye_Docking(
            workdir=openeye_dir,
            pdb_ID=self.protein_pdb,
            mtz=None,
            crystal_path=self.crystal_sdf,
            docking_tuple=docking_tuple
        )
        
        openeye_pipeline.run_oedocking_pipeline(
            n_cpu=self.n_cpus, 
            confs=self.n_confs, 
            mtz=None, 
            mode="oe"
        )
        
        # Collect results
        output_dir = openeye_dir / "output"
        docked_ligands = list(output_dir.glob("*.sdf"))
        
        results_df = self._get_docked_dataframe(docked_ligands, "openeye")
        logger.info(f"OpenEye docking completed with {len(docked_ligands)} results")
        
        return results_df
        
    def _run_rxdock_docking(self, ligands_sdf):
        """Run RxDock docking."""
        logger.info("Starting RxDock docking")
        
        # Create directory for RxDock
        rxdock_dir = self.workdir / "RxDock"
        rxdock_dir.mkdir(exist_ok=True)
        
        # Initialize RxDock docking pipeline
        rxdock_pipeline = RxDock_Docking(
            workdir=rxdock_dir,
            pdb_ID=self.protein_pdb,
            crystal_path=self.crystal_sdf,
            ligands_sdf=ligands_sdf,
            toolkit=self.toolkit
        )
        
        # Run RxDock docking with specified parameters
        rxdock_results = rxdock_pipeline.main(
            n_poses=self.n_confs,
            n_cpus=self.n_cpus
        )
        
        # The RxDock_Docking.main returns a dictionary with results
        if isinstance(rxdock_results, dict) and 'processed_sdf_files' in rxdock_results:
            # Get the list of updated original SDF files with proper molecule naming
            processed_sdf_files = rxdock_results.get('processed_sdf_files', [])
            
            if processed_sdf_files:
                logger.info(f"Processing {len(processed_sdf_files)} RxDock SDF files")
                
                # Use _get_docked_dataframe to process the SDF files
                # This follows the same pattern as plants, gnina, and other methods
                rxdock_df = self._get_docked_dataframe(processed_sdf_files, software="rxdock")
                
                if not rxdock_df.empty:
                    # Add software name if not already present
                    if "Software" not in rxdock_df.columns:
                        rxdock_df["Software"] = "rxdock"
                        
                    # Ensure protein path is included
                    if "Protein_Path" not in rxdock_df.columns:
                        rxdock_df["Protein_Path"] = str(self.protein_pdb.absolute())
                    
                    logger.info(f"RxDock docking completed with {len(rxdock_df)} results")
                    return rxdock_df
            
        # If no results, return empty dataframe
        logger.warning("RxDock docking completed but no results were found")
        return pd.DataFrame()
    
    def _get_docked_dataframe(self, docked_ligands: List[Path], software: str) -> pd.DataFrame:
        """
        Process docked ligands into a dataframe.
        
        Args:
            docked_ligands: List of paths to docked ligand files
            software: Name of the docking software
            
        Returns:
            DataFrame containing processed docking results
        """
        df_list = []
        
        for lig_path in docked_ligands:
            # Load the molecules
            df = PandasTools.LoadSDF(str(lig_path), molColName="Molecule")
            
            # Add the software description and paths
            df["Software"] = software
            df["Protein_Path"] = self.protein_pdb.absolute()
            
            # Software-specific processing
            if software == "plants":
                # For Plants, extract TOTAL_SCORE
                if "TOTAL_SCORE" in df.columns:
                    df = df.rename(columns={"TOTAL_SCORE": "Score"})
                    df_list.append(df)
                else:
                    logger.warning(f"Missing expected column TOTAL_SCORE in Plants results: {lig_path}")
                    
            elif software == "openeye":
                # For OpenEye, extract PLP
                if "PLP" in df.columns:
                    df = df.rename(columns={"PLP": "Score"})
                    df_list.append(df)
                else:
                    logger.warning(f"Missing expected column PLP in OpenEye results: {lig_path}")
                    
            elif software == "gnina":
                # For Gnina, extract CNNaffinity
                if "CNNaffinity" in df.columns:
                    df = df.rename(columns={"CNNaffinity": "Score"})
                    df_list.append(df)
                else:
                    logger.warning(f"Missing expected column CNNaffinity in Gnina results: {lig_path}")
                    
            elif software == "smina":
                # For Smina (same as Gnina but use minimizedAffinity)
                if "minimizedAffinity" in df.columns:
                    df = df.rename(columns={"minimizedAffinity": "Score"})
                    df_list.append(df)
                else:
                    logger.warning(f"Missing expected column minimizedAffinity in Smina results: {lig_path}")
                    
            elif software == "rxdock":
                # For RxDock, the Score column should already be present from the pipeline
                if "SCORE" in df.columns:
                    df = df.rename(columns={"SCORE": "Score"})
                    df_list.append(df)
                else:
                    logger.warning(f"Missing expected column Score in RxDock results: {lig_path}")
        
        # Combine all results
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            logger.warning(f"No valid results found for {software}")
            return pd.DataFrame()
    
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run the docking pipelines for the selected software.
        
        Returns:
            Dictionary mapping software names to result dataframes
        """
        # Process input to SDF if needed
        ligands_sdf = self._process_input_to_sdf()
        
        # Dictionary to store results
        results = {}
        
        # Run each docking software sequentially
        for software in self.docking_software:
            try:
                if software == "plants":
                    results["plants"] = self._run_plants_docking(ligands_sdf)
                    
                elif software == "gnina":
                    gnina_results = self._run_gnina_docking(ligands_sdf)
                    # Add both gnina and smina results
                    for key, value in gnina_results.items():
                        results[key] = value
                    
                elif software == "openeye":
                    results["openeye"] = self._run_openeye_docking(ligands_sdf)
                    
                elif software == "rxdock":
                    results["rxdock"] = self._run_rxdock_docking(ligands_sdf)
            
            except Exception as e:
                logger.error(f"Error running {software} docking: {str(e)}")
                # Continue with next software if one fails
        
        # Store results in instance
        self.results = results
        
        return results 
    
    def concat_df(self) -> pd.DataFrame:
        """
        Concatenate results from different docking software with normalized scores.
        
        Scoring normalization logic:
        1. Each docking software uses a different scoring function with its own scale and direction
        2. 'negative_scoring' indicates whether more negative scores are better (TRUE) or 
           whether higher positive scores are better (FALSE)
        3. All scores are normalized to [0,1] range using MinMaxScaler
        4. If a software uses negative scoring (more negative = better), values are inverted (1-score)
        5. After normalization, ALL scores follow the SAME convention: 
           higher normalized values (closer to 1) = better docking results
           
        Software-specific scoring directions:
        - Gnina: Higher positive scores are better (negative_scoring = FALSE)
        - Smina: Higher positive scores are better (negative_scoring = FALSE)
        - Plants: Higher positive scores are better (negative_scoring = FALSE) [CUSTOM]
        - RxDock: More negative scores are better (negative_scoring = TRUE)
        - OpenEye: More negative scores are better (negative_scoring = TRUE)
        """
        required_cols = ["ID", "Score", "Molecule", "Software", "Protein_Path"]
        df_list = []
        
        for software, df in self.results.items():
            if df.empty:
                continue
                
            # Check required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns {missing_cols} in {software} results")
            
            # Create a copy
            df_copy = df.copy()
            
            # Ensure Score is numeric
            try:
                # First check if we need to convert the values
                if df_copy["Score"].dtype == object:
                    # Try to convert to numeric, forcing errors to NaN
                    df_copy["Score"] = pd.to_numeric(df_copy["Score"], errors="coerce")
                    
                    # Drop rows with NaN scores
                    df_copy = df_copy.dropna(subset=["Score"])
                    
                    if df_copy.empty:
                        print(f"Warning: All scores for {software} were invalid. Skipping.")
                        continue
            except Exception as e:
                print(f"Error processing scores for {software}: {str(e)}")
                continue                # Determine score direction
            if len(df_copy) > 0:
                mean_score = df_copy["Score"].mean()
                negative_scoring = mean_score < 0  # Default: guess direction based on mean
                
                # Override with known scoring directions for each software
                if software.lower() == "gnina":
                    negative_scoring = False  # For Gnina, higher positive scores are better
                
                if software.lower() == "smina":
                    negative_scoring = True  # For Smina, lower negative scores are better
                    
                if software.lower() == "plants":
                    negative_scoring = False  # For Plants, higher positive scores are better
                    
                if software.lower() == "rxdock":
                    negative_scoring = True   # For RxDock, more negative scores are better
                    
                if software.lower() == "openeye":
                    negative_scoring = True   # For OpenEye, more negative PLP scores are better
                
                # Apply scaling
                scaler = MinMaxScaler()
                scores = df_copy[["Score"]].values
                scaled_scores = scaler.fit_transform(scores)
                
                # Invert if negative scoring (more negative = better)
                if negative_scoring:
                    scaled_scores = 1 - scaled_scores  # Invert so that all scores follow the convention: higher normalized values are better
                    
                df_copy["Score"] = scaled_scores
            
            # Select only required columns
            df_selected = df_copy[required_cols]
            df_list.append(df_selected)
        
        # Concatenate all dataframes
        if not df_list:
            return pd.DataFrame(columns=required_cols)
        
        return pd.concat(df_list, ignore_index=True)