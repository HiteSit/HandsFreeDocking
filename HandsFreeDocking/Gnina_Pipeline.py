import os
import subprocess
from pathlib import Path
from typing import List, Tuple

from multiprocessing import Pool
from joblib import Parallel, delayed
from tempfile import gettempdir

from openmm.app import PDBFile
from pdbfixer import PDBFixer

import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import biotite.structure as struc

from rdkit import Chem
from rdkit.Chem import PandasTools

try:
    from openeye import oechem
    from openeye import oeomega
    OPENEYE_AVAILABLE = True
except ImportError:
    OPENEYE_AVAILABLE = False

from .tools.CDPK_Utils import CDPK_Runner, stero_enumerator
from .tools.OpeneEye_Utils import fix_3dmol, get_chirality_and_stereo, gen_3dmol
from .tools.Protein_Preparation import ProteinPreparation_Protoss

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
    def __init__(self, workdir: Path, pdb_ID: Path, crystal_path: Path, ligands_sdf: Path, 
                toolkit: str = "cdpkit", protonation_method: str = "protoss"):
        """
        Initialize the Gnina docking pipeline
        
        Args:
            workdir: Working directory for docking
            pdb_ID: Path to the PDB file
            crystal_path: Path to the crystal ligand file
            ligands_sdf: Path to the ligands SDF file
            toolkit: Which toolkit to use for ligand preparation ("cdpkit" or "openeye")
            protonation_method: Method to protonate the protein ("pdbfixer" or "protoss")
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
        
        # Set the toolkit for ligand preparation
        if toolkit.lower() not in ["cdpkit", "openeye"]:
            raise ValueError(f"Toolkit must be either 'cdpkit' or 'openeye', got {toolkit}")
        
        if toolkit.lower() == "openeye" and not OPENEYE_AVAILABLE:
            logger.warning("OpenEye toolkit not available! Falling back to CDPKit.")
            self.toolkit = "cdpkit"
        else:
            self.toolkit = toolkit.lower()
            
        # Set the protein protonation method
        if protonation_method.lower() not in ["pdbfixer", "protoss"]:
            raise ValueError(f"Protonation method must be either 'pdbfixer' or 'protoss', got {protonation_method}")
        self.protonation_method = protonation_method.lower()

    def _source_macro(self):
        # Grab the protein from the PDB
        protein_path = self.pdb_ID
        reader = pdb.PDBFile.read(protein_path)
        struct_array = reader.get_structure(model=1)

        # Remove all and not protein
        macro_array = struct_array[struc.filter_amino_acids(struct_array)]

        protein_cleaned: Path = self.workdir / f"{self.pdb_ID.stem}_clean.pdb"
        self.protein_cleaned: Path = protein_cleaned

        strucio.save_structure(str(protein_cleaned), macro_array)

    def prepare_protein(self) -> None:
        # Define the output path for the prepared protein
        protein_prepared: Path = self.workdir / f"{self.pdb_ID.stem}_prep.pdb"
        self.protein_prepared = protein_prepared.absolute()
        
        if self.protonation_method == "pdbfixer":
            # Use PDBFixer for protonation
            fixer = PDBFixer(filename=str(self.protein_cleaned))
            fixer.removeHeterogens(True)
            fixer.addMissingHydrogens(7.0)
            
            # Save the prepared protein
            PDBFile.writeFile(fixer.topology, fixer.positions, open(str(protein_prepared), 'w'), keepIds=True)
        
        elif self.protonation_method == "protoss":
            # Use Protoss for protonation
            protoss = ProteinPreparation_Protoss()
            protoss(self.protein_cleaned, protein_prepared)

    def prepare_ligands(self):
        ligands_splitted_path: Path = self.workdir / "ligands_split"
        ligands_splitted_path.mkdir(exist_ok=True)
        
        if self.toolkit == "cdpkit":
            # For CDPKit workflow:
            # 1. First enumerate stereoisomers using RDKit-based function
            ligands_stereo_path = self.workdir / f"{self.ligands_sdf.stem}_stereo.sdf"
            logger.info(f"Enumerating stereoisomers with RDKit for {self.ligands_sdf}")
            ligands_stereo_path = stero_enumerator(self.ligands_sdf, ligands_stereo_path)
            
            # 2. Then prepare the ligands using CDPK
            ligand_prepared_path = self.workdir / "ligands_prepared.sdf"
            logger.info(f"Preparing ligands with CDPKit")
            cdpk_runner = CDPK_Runner()
            cdpk_runner.prepare_ligands(ligands_stereo_path, ligand_prepared_path)
            
            # 3. Split into individual files
            logger.info(f"Splitting prepared ligands into individual files")
            for mol in Chem.SDMolSupplier(str(ligand_prepared_path)):
                mol_name = mol.GetProp("_Name")
                ligand_split = ligands_splitted_path / f"{mol_name}.sdf"
                
                self.ligands_splitted.append(ligand_split.absolute())
                Chem.SDWriter(str(ligand_split)).write(mol)
        else:
            # OpenEye method for ligand preparation
            # Get SMILES from SDF file first to use with gen_3dmol
            logger.info(f"Extracting SMILES from SDF file to prepare with OpenEye toolkit")
            molecules_data = []
            
            ifs = oechem.oemolistream()
            if not ifs.open(str(self.ligands_sdf)):
                raise FileNotFoundError(f"Unable to open {self.ligands_sdf}")
                
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
                    # Use 'Iso' naming to be consistent with CDPKit pattern across pipelines
                    enantiomer_name = f"{title}_Iso{j}"
                    enantiomer.SetTitle(enantiomer_name)
                    
                    # Get and store chirality information
                    chirality_info = get_chirality_and_stereo(enantiomer)
                    if chirality_info:
                        oechem.OESetSDData(enantiomer, "ChiralInfo", chirality_info)
                    
                    # Save to SDF file
                    ligand_split = ligands_splitted_path / f"{enantiomer_name}.sdf"
                    self.ligands_splitted.append(ligand_split.absolute())
                    
                    ofs = oechem.oemolostream(str(ligand_split))
                    oechem.OEWriteMolecule(ofs, enantiomer)
                    ofs.close()

    @staticmethod
    def add_pose_num(lig_docked: Path):
        # Load molecules from the input SDF file
        supplier = Chem.SDMolSupplier(str(lig_docked))
        molecules = list(supplier)
        
        # Overwrite the same file by writing back to lig_docked
        writer = Chem.SDWriter(str(lig_docked))
        
        for i, mol in enumerate(molecules):
            num = i + 1
            name = mol.GetProp("_Name")
            if name.startswith("_"):
                name = name[1:]
            new_name = f"{name}_Gnina-P{num}"
            mol.SetProp("_Name", new_name)
            writer.write(mol)
        
        writer.close()

    def _prepare_gnina_command(self, ligand_split: Path, output_ligand: Path,
                               n_confs: int, n_cpu: int,
                               atom_to_covalent: str = None, smarts_react: str = None) -> List[str]:
        if atom_to_covalent and smarts_react:
            output_ligand_flex = self.docked_final_dir / f"{ligand_split.stem}_Gnina_Flex.sdf"
            output_ligand_flex = output_ligand_flex.absolute()
            self.docked_gnina_flex.append(output_ligand_flex)
            return [
                'gnina',
                '-r', str(self.protein_prepared),
                '-l', str(ligand_split),
                '--autobox_ligand', str(self.crystal_path),
                '--covalent_rec_atom', atom_to_covalent,
                '--covalent_lig_atom_pattern', f'[$({smarts_react})]',
                '--covalent_bond_order', '1',
                '--pose_sort_order', 'CNNaffinity',
                '--exhaustiveness', '32',
                '--num_modes', str(n_confs),
                "--min_rmsd_filter", "1.5",
                "--cpu", str(n_cpu),
                "--covalent_optimize_lig",
                "--full_flex_output",
                '--out_flex', str(output_ligand_flex), 
                "-o", str(output_ligand),
                "--no_gpu"
            ]
        else:
            return [
                "gnina",
                f"--receptor", str(self.protein_prepared),
                f"--ligand", str(ligand_split),
                f"--autobox_ligand", str(self.crystal_path),
                f"--out", str(output_ligand),
                "--cpu", str(n_cpu),
                "--num_modes", str(n_confs),
                "--no_gpu"
            ]

    def _run_gnina_commands(self, gnina_commands: List[List[str]]):
        def runner(command_lst: List[str]):
            log_file_path = self.workdir / "gnina_commands.log"
            with open(log_file_path, "a") as log_file:
                subprocess.run(command_lst, check=True, text=True, cwd=self.workdir, stdout=log_file, stderr=log_file)

        logger.info(f"Running {len(gnina_commands)} gnina commands")
        for command in gnina_commands:
            runner(command)

    def non_covalent_run(self, n_confs: int, n_cpus: int):
        self._source_macro()
        self.prepare_protein()
        self.prepare_ligands()

        gnina_commands: List = []
        docked_ligands: List[Path] = []
        for ligand_split in self.ligands_splitted:
            output_ligand = self.docked_final_dir / f"{ligand_split.stem}_Gnina.sdf"
            output_ligand = output_ligand.absolute()
            self.docked_gnina.append(output_ligand)

            gnina_command = self._prepare_gnina_command(ligand_split, output_ligand, n_confs, n_cpus)
            gnina_commands.append(gnina_command)
            docked_ligands.append(output_ligand)

        self._run_gnina_commands(gnina_commands)
        [self.add_pose_num(lig_docked) for lig_docked in docked_ligands]

    def covalent_run(self, n_confs: int, n_cpus: int, atom_to_covalent: str, smarts_react: str):
        self._source_macro()
        self.prepare_protein()
        self.prepare_ligands()

        gnina_commands: List = []
        docked_ligands: List[Path] = []
        for ligand_split in self.ligands_splitted:
            output_ligand = self.docked_final_dir / f"{ligand_split.stem}_Gnina.sdf"
            output_ligand = output_ligand.absolute()
            self.docked_gnina.append(output_ligand)

            gnina_command = self._prepare_gnina_command(ligand_split, output_ligand, n_confs, n_cpus, atom_to_covalent, smarts_react)
            gnina_commands.append(gnina_command)
            docked_ligands.append(output_ligand)

        self._run_gnina_commands(gnina_commands)
        [self.add_pose_num(lig_docked) for lig_docked in docked_ligands]

class Convert_Gnina:
    def __init__(self):
        pass
