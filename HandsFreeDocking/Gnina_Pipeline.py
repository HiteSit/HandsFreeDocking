"""
Gnina Pipeline Module for Molecular Docking

This module provides a comprehensive pipeline for molecular docking using the 
Gnina docking software. It supports both covalent and non-covalent docking protocols
with flexible options for ligand preparation using either CDPKit or OpenEye toolkits,
and protein preparation using either PDBFixer or Protoss methods.

The main class, Gnina_Docking, handles the entire docking workflow including:
1. Protein preparation and cleaning
2. Ligand preparation with stereoisomer enumeration
3. Docking with Gnina
4. Processing and organizing docking results

Key features:
- Support for both covalent and non-covalent docking
- Multiple options for ligand preparation (CDPKit or OpenEye)
- Multiple options for protein preparation (PDBFixer or Protoss)
- Parallel processing for improved performance
- Handling of stereoisomers
- Structured output organization

Dependencies:
- Gnina: External molecular docking software
- RDKit: For basic molecular operations
- OpenEye (optional): For advanced ligand preparation
- CDPKit: For ligand preparation
- PDBFixer and OpenMM: For protein preparation
- Biotite: For protein structure manipulation
- Protoss: For protein protonation (optional)
"""

# Standard library imports
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool
from tempfile import gettempdir

# Third-party computational chemistry imports
# Protein preparation
from openmm.app import PDBFile
from pdbfixer import PDBFixer

# Biotite for protein structure manipulation
import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import biotite.structure as struc

# RDKit for cheminformatics
from rdkit import Chem
from rdkit.Chem import PandasTools

# Parallel processing
from joblib import Parallel, delayed

# Optional OpenEye imports
try:
    from openeye import oechem
    from openeye import oeomega
    OPENEYE_AVAILABLE = True
except ImportError:
    OPENEYE_AVAILABLE = False

# Local imports
from .tools.Ligand_Preparation import LigandPreparator
from .tools.Protein_Preparation import ProteinPreparation_Protoss

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Gnina_Docking:
    """
    Gnina molecular docking pipeline for both covalent and non-covalent docking
    
    This class implements a complete workflow for molecular docking using the Gnina
    software. It handles protein preparation, ligand preparation, docking, and result
    processing. The pipeline supports multiple toolkits for ligand preparation and 
    protein protonation to accommodate different preferences and license availability.
    
    Attributes:
        workdir (Path): Working directory for all docking operations and output files
        pdb_ID (Path): Path to the input protein PDB file
        crystal_path (Path): Path to the crystal (reference) ligand used for binding site definition
        ligands_sdf (Path): Path to the input ligands in SDF format
        toolkit (str): Toolkit used for ligand preparation ('cdpkit' or 'openeye')
        protonation_method (str): Method used for protein protonation ('pdbfixer' or 'protoss')
        protein_cleaned (Path): Path to the cleaned protein
        protein_prepared (Path): Path to the fully prepared protein ready for docking
        ligands_splitted (List[Path]): List of paths to individual prepared ligands
        docked_final_dir (Path): Directory for final docking outputs
        docked_gnina (List[Path]): List of paths to docked ligand output files
        docked_gnina_flex (List[Path]): List of paths to flexible docked ligand output files (for covalent docking)
    """
    
    def __init__(self, workdir: Path, pdb_ID: Path, crystal_path: Path, ligands_sdf: Path, 
                protonation_method: str = "cdp", protein_protonation_method: str = "protoss", 
                tautomer_score_threshold: Optional[float] = None):
        """
        Initialize the Gnina docking pipeline with all necessary parameters and directories
        
        Args:
            workdir (Path): Working directory for all docking operations and output files
            pdb_ID (Path): Path to the input protein PDB file
            crystal_path (Path): Path to the crystal (reference) ligand used for binding site definition
            ligands_sdf (Path): Path to the input ligands in SDF format
            protonation_method (str): Method for protonating ligands ("cdp", "oe", or "scrubber")
            protein_protonation_method (str): Method to protonate the protein, options:
                                     - "protoss": Use Protoss (default, requires license)
                                     - "pdbfixer": Use PDBFixer (open source)
            tautomer_score_threshold: Score threshold for tautomer selection (None = best only, value = list within threshold)
        
        Raises:
            ValueError: If an invalid protonation method is specified
        """
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True)

        self.pdb_ID = pdb_ID
        self.crystal_path = crystal_path.absolute()
        self.ligands_sdf: Path = ligands_sdf.absolute()

        # Added in the class
        self.protein_cleaned: Path = Path()
        self.protein_prepared: Path = Path()
        self.ligands_splitted: List[Path] = []

        self.docked_final_dir: Path = workdir / "output"
        self.docked_final_dir.mkdir(exist_ok=True)

        self.docked_gnina: List[Path] = []
        self.docked_gnina_flex: List[Path] = []
        
        # Set the ligand protonation method
        if protonation_method.lower() not in ["cdp", "oe", "scrubber"]:
            raise ValueError(f"Ligand protonation method must be 'cdp', 'oe', or 'scrubber', got {protonation_method}")
        self.protonation_method = protonation_method.lower()
        self.tautomer_score_threshold = tautomer_score_threshold
            
        # Set the protein protonation method
        if protein_protonation_method.lower() not in ["pdbfixer", "protoss"]:
            raise ValueError(f"Protein protonation method must be either 'pdbfixer' or 'protoss', got {protein_protonation_method}")
        self.protein_protonation_method = protein_protonation_method.lower()

    def _source_macro(self):
        """
        Extract and clean the protein structure from the PDB file
        
        This private method reads the input PDB file, extracts only the protein
        component (amino acids), and saves it as a cleaned PDB file. It filters
        out water molecules, ligands, and other non-protein components.
        
        The cleaned protein file is saved with '_clean.pdb' suffix and the path
        is stored in self.protein_cleaned for further processing.
        
        Returns:
            None
        """
        # Grab the protein from the PDB file
        protein_path = self.pdb_ID
        reader = pdb.PDBFile.read(protein_path)
        struct_array = reader.get_structure(model=1)

        # Remove all non-protein atoms/residues using biotite's filter
        macro_array = struct_array[struc.filter_amino_acids(struct_array)]

        # Define output path for the cleaned protein
        protein_cleaned: Path = self.workdir / f"{self.pdb_ID.stem}_clean.pdb"
        self.protein_cleaned: Path = protein_cleaned

        # Save the cleaned protein structure
        strucio.save_structure(str(protein_cleaned), macro_array)

    def prepare_protein(self) -> None:
        """
        Prepare the protein structure for docking by adding hydrogens and optimizing protonation states
        
        This method takes the cleaned protein (produced by _source_macro) and prepares it
        for docking by adding hydrogens and assigning proper protonation states. Two methods
        are supported:
        1. PDBFixer: An open-source tool from OpenMM suite, uses a simple pH-based model
        2. Protoss: A commercial tool with more advanced protonation state prediction
        
        The method used is determined by the self.protonation_method attribute set during initialization.
        The prepared protein is saved with a '_prep.pdb' suffix and the path is stored in
        self.protein_prepared for use in docking.
        
        Returns:
            None
        """
        # Define the output path for the prepared protein
        protein_prepared: Path = self.workdir / f"{self.pdb_ID.stem}_prep.pdb"
        self.protein_prepared = protein_prepared.absolute()
        
        if self.protein_protonation_method == "pdbfixer":
            # Use PDBFixer for protonation - open source method
            logger.info(f"Preparing protein using PDBFixer at pH 7.0")
            fixer = PDBFixer(filename=str(self.protein_cleaned))
            fixer.removeHeterogens(True)  # Remove any remaining non-protein components
            fixer.addMissingHydrogens(7.0)  # Add hydrogens at pH 7.0
            
            # Save the prepared protein
            PDBFile.writeFile(fixer.topology, fixer.positions, open(str(protein_prepared), 'w'), keepIds=True)
        
        elif self.protonation_method == "protoss":
            # Use Protoss for protonation - commercial method with more advanced features
            logger.info(f"Preparing protein using Protoss")
            protoss = ProteinPreparation_Protoss()
            protoss(self.protein_cleaned, protein_prepared)

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

    @staticmethod
    def add_pose_num(lig_docked: Path) -> None:
        """
        Add pose numbers to docked ligand names for better identification
        
        After docking, this static method processes the output SDF file to add pose numbers
        to each molecule's name. This makes it easier to identify and track different
        binding poses in downstream analysis. The naming convention uses "Gnina-P{num}" suffix
        where num is the sequential pose number (starting from 1).
        
        Args:
            lig_docked (Path): Path to the docked ligand SDF file to process
            
        Returns:
            None
        
        Note:
            This method modifies the file in-place by overwriting it with the updated names.
        """
        # Load molecules from the input SDF file
        supplier = Chem.SDMolSupplier(str(lig_docked))
        molecules = list(supplier)
        
        # Overwrite the same file by writing back to lig_docked
        writer = Chem.SDWriter(str(lig_docked))
        
        for i, mol in enumerate(molecules):
            num = i + 1  # Pose numbers start at 1 rather than 0
            name = mol.GetProp("_Name")
            # Remove leading underscore if present
            if name.startswith("_"):
                name = name[1:]
            # Create new name with pose number
            new_name = f"{name}_Gnina-P{num}"
            mol.SetProp("_Name", new_name)
            writer.write(mol)
        
        writer.close()

    def _prepare_gnina_command(self, ligand_split: Path, output_ligand: Path,
                               n_confs: int, n_cpu: int,
                               atom_to_covalent: str = None, smarts_react: str = None) -> List[str]:
        """
        Prepare the Gnina docking command with appropriate arguments
        
        This method constructs the command-line arguments for the Gnina docking program.
        It handles both standard non-covalent docking and covalent docking scenarios
        depending on the provided parameters.
        
        Args:
            ligand_split (Path): Path to the prepared individual ligand file
            output_ligand (Path): Path where docked poses should be saved
            n_confs (int): Number of conformations/poses to generate
            n_cpu (int): Number of CPU cores to use for docking
            atom_to_covalent (str, optional): Atom ID in receptor for covalent docking
            smarts_react (str, optional): SMARTS pattern for reactive group in ligand
            
        Returns:
            List[str]: Command line arguments list for subprocess call
            
        Note:
            If atom_to_covalent and smarts_react are both provided, covalent docking
            will be performed with additional parameters. Otherwise, standard
            non-covalent docking is performed.
        """
        if atom_to_covalent and smarts_react:
            # Covalent docking mode - requires special parameters
            # Create a separate output file for flexible receptor output
            output_ligand_flex = self.docked_final_dir / f"{ligand_split.stem}_Gnina_Flex.sdf"
            output_ligand_flex = output_ligand_flex.absolute()
            self.docked_gnina_flex.append(output_ligand_flex)
            
            # Full command for covalent docking
            return [
                'gnina',  # Gnina executable
                '-r', str(self.protein_prepared),  # Receptor (protein)
                '-l', str(ligand_split),  # Ligand
                '--autobox_ligand', str(self.crystal_path),  # Reference ligand for binding site
                '--covalent_rec_atom', atom_to_covalent,  # Receptor atom for covalent bond
                '--covalent_lig_atom_pattern', f'[$({smarts_react})]',  # Reactive group in ligand
                '--covalent_bond_order', '1',  # Single bond covalent attachment
                '--pose_sort_order', 'CNNaffinity',  # Sort poses by CNN affinity score
                '--exhaustiveness', '32',  # Thoroughness of search (higher = more thorough)
                '--num_modes', str(n_confs),  # Number of binding modes to generate
                "--min_rmsd_filter", "1.5",  # Minimum RMSD between output poses
                "--cpu", str(n_cpu),  # Number of CPU cores to use
                "--covalent_optimize_lig",  # Optimize ligand positions for covalent bonding
                "--full_flex_output",  # Output full flexible receptor
                '--out_flex', str(output_ligand_flex),  # Path for flexible receptor output
                "-o", str(output_ligand),  # Path for docked poses output
                # "--no_gpu"  # Don't use GPU (CPU only)
            ]
        else:
            # Standard non-covalent docking with simpler parameters
            return [
                "gnina",  # Gnina executable
                f"--receptor", str(self.protein_prepared),  # Receptor (protein)
                f"--ligand", str(ligand_split),  # Ligand
                f"--autobox_ligand", str(self.crystal_path),  # Reference ligand for binding site
                f"--out", str(output_ligand),  # Path for docked poses output
                "--cpu", str(n_cpu),  # Number of CPU cores to use
                "--num_modes", str(n_confs),  # Number of binding modes to generate
                # "--no_gpu"  # Don't use GPU (CPU only)
            ]

    def _run_gnina_commands(self, gnina_commands: List[List[str]]) -> None:
        """
        Execute the prepared Gnina docking commands sequentially
        
        This method takes a list of prepared Gnina command line argument lists and
        executes them one by one. The output of each command is logged to a file for
        reference and debugging purposes.
        
        Args:
            gnina_commands (List[List[str]]): List of command line argument lists
                                              prepared by _prepare_gnina_command
        
        Returns:
            None
            
        Note:
            All command outputs (stdout and stderr) are appended to gnina_commands.log
            in the working directory.
        """
        def runner(command_lst: List[str]):
            """
            Helper function to run a single command and log its output
            
            Args:
                command_lst (List[str]): Command line arguments to execute
            """
            log_file_path = self.workdir / "gnina_commands.log"
            with open(log_file_path, "a") as log_file:
                env = os.environ.copy()
                env["PATH"] = "/home/hitesit/Software/gnina:" + env.get("PATH", "")
                subprocess.run(command_lst, check=True, text=True, cwd=self.workdir, stdout=log_file, stderr=log_file, env=env)

        logger.info(f"Running {len(gnina_commands)} gnina commands")
        for command in gnina_commands:
            runner(command)

    def non_covalent_run(self, n_confs: int, n_cpus: int) -> None:
        """
        Execute the complete non-covalent docking pipeline
        
        This method runs the entire docking workflow for standard, non-covalent docking:
        1. Clean and prepare the protein structure
        2. Prepare the ligands with 3D conformers
        3. Generate docking commands for each ligand
        4. Execute all docking commands
        5. Process the results to add pose numbering
        
        Args:
            n_confs (int): Number of conformations to generate per ligand
            n_cpus (int): Number of CPU cores to use for each docking run
            
        Returns:
            None
            
        Note:
            All docked structures will be saved in the output directory with
            filenames following the pattern: {ligand_name}_Gnina.sdf
        """
        # Step 1: Clean the protein (extract only amino acids)
        self._source_macro()
        # Step 2: Prepare the protein (add hydrogens, optimize protonation)
        self.prepare_protein()
        # Step 3: Prepare ligands (generate 3D, enumerate stereoisomers)
        self.prepare_ligands()

        # Step 4: Generate docking commands for each ligand
        gnina_commands: List = []
        docked_ligands: List[Path] = []
        for ligand_split in self.ligands_splitted:
            # Define output file path for this ligand
            output_ligand = self.docked_final_dir / f"{ligand_split.stem}_Gnina.sdf"
            output_ligand = output_ligand.absolute()
            self.docked_gnina.append(output_ligand)

            # Prepare the gnina command for this ligand
            gnina_command = self._prepare_gnina_command(ligand_split, output_ligand, n_confs, n_cpus)
            gnina_commands.append(gnina_command)
            docked_ligands.append(output_ligand)

        # Step 5: Run all docking commands
        self._run_gnina_commands(gnina_commands)
        # Step 6: Process results to add pose numbering
        [self.add_pose_num(lig_docked) for lig_docked in docked_ligands]

    def covalent_run(self, n_confs: int, n_cpus: int, atom_to_covalent: str, smarts_react: str) -> None:
        """
        Execute the complete covalent docking pipeline
        
        This method runs the entire docking workflow for covalent docking, where a covalent
        bond is formed between a specific protein atom and a reactive group in the ligand:
        1. Clean and prepare the protein structure
        2. Prepare the ligands with 3D conformers
        3. Generate covalent docking commands for each ligand
        4. Execute all docking commands
        5. Process the results to add pose numbering
        
        Args:
            n_confs (int): Number of conformations to generate per ligand
            n_cpus (int): Number of CPU cores to use for each docking run
            atom_to_covalent (str): Atom ID in the receptor to form the covalent bond
                                    (format: "chainID:resID:atomName", e.g., "A:145:SG")
            smarts_react (str): SMARTS pattern defining the reactive group in ligands
                               (e.g., "[C;H1,H2]=O" for aldehydes)
            
        Returns:
            None
            
        Note:
            All docked structures will be saved in the output directory with
            filenames following the pattern: {ligand_name}_Gnina.sdf
            Flexible receptor outputs will be saved as {ligand_name}_Gnina_Flex.sdf
        """
        # Step 1: Clean the protein (extract only amino acids)
        self._source_macro()
        # Step 2: Prepare the protein (add hydrogens, optimize protonation)
        self.prepare_protein()
        # Step 3: Prepare ligands (generate 3D, enumerate stereoisomers)
        self.prepare_ligands()

        # Step 4: Generate covalent docking commands for each ligand
        gnina_commands: List = []
        docked_ligands: List[Path] = []
        for ligand_split in self.ligands_splitted:
            # Define output file path for this ligand
            output_ligand = self.docked_final_dir / f"{ligand_split.stem}_Gnina.sdf"
            output_ligand = output_ligand.absolute()
            self.docked_gnina.append(output_ligand)

            # Prepare the gnina command for covalent docking
            # Passing atom_to_covalent and smarts_react triggers covalent mode
            gnina_command = self._prepare_gnina_command(ligand_split, output_ligand, n_confs, n_cpus, atom_to_covalent, smarts_react)
            gnina_commands.append(gnina_command)
            docked_ligands.append(output_ligand)

        # Step 5: Run all docking commands
        self._run_gnina_commands(gnina_commands)
        # Step 6: Process results to add pose numbering
        [self.add_pose_num(lig_docked) for lig_docked in docked_ligands]

class Convert_Gnina:
    """
    Utility class for converting Gnina docking results to other formats
    
    This class is a placeholder for implementing conversion utilities between
    Gnina output formats and other widely used molecular formats or
    visualization tools.
    
    Note:
        Currently implemented as a stub for future expansion.
    """
    
    def __init__(self):
        """
        Initialize the Convert_Gnina utility class
        
        Note:
            Currently a placeholder for future implementation.
        """
        pass
