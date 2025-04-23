import io
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from urllib.parse import urljoin
import tempfile
import glob

import requests
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ProteinPreparation_Protoss:
    """
    A class for preparing protein structures using ProtoSS service.
    """
    
    def __init__(self):
        """Initialize the ProteinPreparation class with API endpoints."""
        self.PROTEINS_PLUS_URL = 'https://proteins.plus/api/v2/'
        self.UPLOAD = urljoin(self.PROTEINS_PLUS_URL, 'molecule_handler/upload/')
        self.UPLOAD_JOBS = urljoin(self.PROTEINS_PLUS_URL, 'molecule_handler/upload/jobs/')
        self.PROTEINS = urljoin(self.PROTEINS_PLUS_URL, 'molecule_handler/proteins/')
        self.LIGANDS = urljoin(self.PROTEINS_PLUS_URL, 'molecule_handler/ligands/')
        self.PROTOSS = urljoin(self.PROTEINS_PLUS_URL, 'protoss/')
        self.PROTOSS_JOBS = urljoin(self.PROTEINS_PLUS_URL, 'protoss/jobs/')
    
    def poll_job(self, job_id, poll_url, poll_interval=1, max_polls=10):
        """
        Poll the progress of a job by continuously polling the server in regular intervals and updating the job information.

        Args:
            job_id (str): UUID of the job to poll.
            poll_url (str): URL to send the polling request to.
            poll_interval (int): Time interval between polls in seconds. Default is 1 second.
            max_polls (int): Maximum number of times to poll before exiting. Default is 10.

        Returns:
            dict: Polled job information.
        """
        # Get the initial job information
        job = requests.get(poll_url + job_id + '/').json()
        status = job['status']
        current_poll = 0

        # Continuously poll the job until it is completed or maximum polls reached
        while status == 'pending' or status == 'running':
            print(f'Job {job_id} is {status}')
            current_poll += 1

            # Check if maximum polls reached
            if current_poll >= max_polls:
                print(f'Job {job_id} has not completed after {max_polls} polling requests and {poll_interval * max_polls} seconds')
                return job

            # Wait for the specified interval before polling again
            time.sleep(poll_interval)

            # Poll the job again to get updated status
            job = requests.get(poll_url + job_id + '/').json()
            status = job['status']

        print(f'Job {job_id} completed with {status}')
        return job

    def prepare_protein_protoss(self, input_pdb: Path, output_pdb: Path) -> Path:
        """
        Prepares a protein using ProtoSS.

        Args:
            input_pdb (Path): Path to the input protein file in PDB format.
            output_pdb (Path): Path to save the prepared protein file.

        Returns:
            Path: Path to the prepared protein file in PDB format.
        """
        # Print log message
        print('Preparing protein with ProtoSS ...')

        # Convert CIF to PDB if needed
        temp_pdb = None
        if input_pdb.suffix.lower() == '.cif':
            print('Converting CIF to PDB format...')
            temp_pdb = Path(tempfile.mktemp(suffix='.pdb'))
            
            # Read the CIF file
            cif_file = pdbx.CIFFile.read(str(input_pdb))
            
            # Get the structure from the CIF file
            structure = pdbx.get_structure(cif_file, model=1)  # Get the first model
            
            # Create a PDB file object and set the structure
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(structure)
            
            # Write the PDB file
            pdb_file.write(str(temp_pdb))
            
            # Use the temporary PDB file for processing
            input_pdb_for_processing = temp_pdb
            print(f'Converted CIF to PDB: {temp_pdb}')
        else:
            input_pdb_for_processing = input_pdb

        # Open the receptor protein file
        with open(input_pdb_for_processing) as upload_file:
            # Create the query with the protein file
            query = {'protein_file': upload_file}
            # Submit the job to ProtoSS and get the job submission response
            job_submission = requests.post(self.PROTOSS, files=query).json()

        # Poll the job status until it is completed
        protoss_job = self.poll_job(job_submission.get('job_id'), self.PROTOSS_JOBS)

        # Get the output protein information from the job
        protossed_protein = requests.get(self.PROTEINS + protoss_job['output_protein'] + '/').json()

        # Create a StringIO object with the protein file string
        protein_file = io.StringIO(protossed_protein['file_string'])

        # Parse the protein structure from the StringIO object
        protein_structure = PDBParser().get_structure(protossed_protein['name'], protein_file)
        
        # Ensure the output directory exists
        output_pdb.parent.mkdir(parents=True, exist_ok=True)

        # Open the output file in write mode
        with output_pdb.open('w') as output_file_handle:
            # Create a PDBIO object
            pdbio = PDBIO()
            # Set the protein structure for saving
            pdbio.set_structure(protein_structure)
            # Save the protein structure to the output file
            pdbio.save(output_file_handle)
        
        # Clean up temporary file if created
        if temp_pdb and temp_pdb.exists():
            temp_pdb.unlink()
        
        # Return the path to the prepared protein file
        return output_pdb
    
    def __call__(self, input_pdb: Path, output_pdb: Path) -> Path:
        """
        Call method that wraps prepare_protein_protoss for easier usage.
        
        Args:
            input_pdb (Path): Path to the input protein file in PDB or CIF format.
            output_pdb (Path): Path to save the prepared protein file.
            
        Returns:
            Path: Path to the prepared protein file in PDB format.
        """
        return self.prepare_protein_protoss(input_pdb, output_pdb)

class ProteinPreparation_PDBFixer:
    """
    A class for preparing protein structures using PDBFixer.
    """
    
    def __call__(self, input_pdb: Path, output_pdb: Path) -> Path:
        """
        Call method that wraps prepare_protein_pdb_fixer for easier usage.
        """
        
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        fixer = PDBFixer(filename=str(input_pdb))
        fixer.removeHeterogens(True)

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.findNonstandardResidues()

        fixer.addMissingHydrogens(7.0)

        PDBFile.writeFile(fixer.topology, fixer.positions, open(str(output_pdb), 'w'), keepIds=True)
        return output_pdb


class ProteinPreparation_Meeko:
    """
    A class for preparing protein structures using Meeko.
    This class can handle both standard protein preparation and preparation with a crystal ligand.
    """
    
    def __init__(self):
        """
        Initialize the ProteinPreparation_Meeko class.
        """
        # Hardcoded environment variable path for mmtbx.reduce2
        # This is a placeholder and can be easily removed or modified
        self.mmtbx_ccp4_monomer_lib = "/home/hitesit/Software/FORK/geostd"
        
        # Error tracking attributes
        self.stdout_reduce = None
        self.stderr_reduce = None
        self.success_reduce = None
        
        self.stdout_meeko = None
        self.stderr_meeko = None
        self.success_meeko = None
        
        self.prody_error = None
        self.pymol_error = None
    
    def prepare_protein_meeko(self, input_pdb: Path, output_pdb: Path, crystal_ligand_sdf: Path = None) -> dict:
        """
        Prepares a protein using Meeko.
        
        Args:
            input_pdb (Path): Path to the input protein file in PDB format.
            output_pdb (Path): Path to save the prepared protein file.
            crystal_ligand_sdf (Path, optional): Path to the crystal ligand in SDF format.
                                               If provided, box dimensions will be calculated.
        
        Returns:
            dict or Path: If crystal_ligand_sdf is provided, returns a dictionary with paths and contents.
                         Otherwise, returns the path to the prepared protein file in PDBQT format.
        """
        import os
        import subprocess
        import tempfile
        from pathlib import Path
        import traceback
        
        # Reset error tracking attributes for this run
        self.stdout_reduce = None
        self.stderr_reduce = None
        self.success_reduce = False
        
        self.stdout_meeko = None
        self.stderr_meeko = None
        self.success_meeko = False
        
        self.prody_error = None
        self.pymol_error = None
        
        # Create temporary directory and files
        temp_dir = tempfile.gettempdir()
        temp_hydrogenated = Path(tempfile.mktemp(suffix="_meeko_H.pdb"))
        temp_aligned = Path(tempfile.mktemp(suffix="_aligned.pdb"))
        temp_protein = Path(tempfile.mktemp(suffix="_protein.pdb"))
        
        # Set environment variables for mmtbx.reduce2
        env = os.environ.copy()
        env["MMTBX_CCP4_MONOMER_LIB"] = self.mmtbx_ccp4_monomer_lib
        
        # Step 1: Add hydrogens using mmtbx.reduce2
        result = subprocess.run(
            [
                "mmtbx.reduce2",
                str(input_pdb),
                "approach=add", "add_flip_movers=True", f"output.filename={temp_hydrogenated}", "--overwrite"
            ],
            capture_output=True,
            text=True,
            env=env
        )
        
        self.stdout_reduce = result.stdout
        self.stderr_reduce = result.stderr
        self.success_reduce = result.returncode == 0
        
        if not self.success_reduce:
            # Even if reduce2 fails, we'll try to continue if the output file exists
            if not temp_hydrogenated.exists():
                # If the file doesn't exist, we can't continue
                return {
                    "pdbqt_path": None,
                    "box_pdb": None,
                    "box_txt": None
                } if crystal_ligand_sdf else None
        
        # Step 2: Align the hydrogenated structure with the original using PyMOL
        try:
            from prody import parsePDB, writePDB, calcCenter
            from pymol import cmd
            
            cmd.reinitialize()
            cmd.load(str(input_pdb), "Protein")
            cmd.load(str(temp_hydrogenated), "Protein_H")
            cmd.align("Protein_H", "Protein")
            
            # If crystal ligand is provided, load it and create a combined structure
            if crystal_ligand_sdf:
                cmd.load(str(crystal_ligand_sdf), "Crystal")
                cmd.create("Protein_Crystal", "Protein_H Crystal")
                cmd.save(str(temp_aligned), "Protein_Crystal")
                
                # Parse the aligned structure to extract protein and ligand atoms
                atoms = parsePDB(str(temp_aligned))
                receptor_atoms = atoms.select("protein and not water and not hetero")
                ligand_atoms = atoms.select("not protein and not water")
                
                # Write protein atoms to a temporary file
                writePDB(str(temp_protein), receptor_atoms)
                
                # Calculate box center and dimensions based on the ligand
                center_x, center_y, center_z = calcCenter(ligand_atoms)
                padding_x, padding_y, padding_z = (10, 10, 10)  # Default padding
                
                # Ensure output directory exists
                output_pdb.parent.mkdir(parents=True, exist_ok=True)
                
                # Step 3: Run mk_prepare_receptor.py with box parameters
                result = subprocess.run(
                    [
                        "mk_prepare_receptor.py",
                        "-i", str(temp_protein),
                        "-o", str(output_pdb.with_suffix('')),  # Remove suffix to use as base name
                        "-p", "-v",
                        "--box_center",
                        str(center_x), str(center_y), str(center_z),
                        "--box_size",
                        str(padding_x), str(padding_y), str(padding_z),
                        "--allow_bad_res",
                    ],
                    capture_output=True,
                    text=True
                )
                
                self.stdout_meeko = result.stdout
                self.stderr_meeko = result.stderr
                self.success_meeko = result.returncode == 0
                
                # Read the generated files into variables
                pdbqt_path = output_pdb.with_suffix('.pdbqt')
                box_pdb_path = output_pdb.with_suffix('.box.pdb')
                box_txt_path = output_pdb.with_suffix('.box.txt')
                
                # Create a dictionary with file paths and contents
                result_dict = {
                    "pdbqt_path": str(pdbqt_path) if pdbqt_path.exists() else None,
                    "box_pdb": box_pdb_path.read_text() if box_pdb_path.exists() else None,
                    "box_txt": box_txt_path.read_text() if box_txt_path.exists() else None
                }
                
                # Delete the box files after reading their contents
                if box_pdb_path.exists():
                    box_pdb_path.unlink()
                if box_txt_path.exists():
                    box_txt_path.unlink()
                
                # Clean up temporary files
                self._cleanup_temp_files([temp_hydrogenated, temp_aligned, temp_protein])
                
                return result_dict
            else:
                # If no crystal ligand is provided, just save the aligned protein
                cmd.save(str(temp_aligned), "Protein_H")
                
                # Parse the aligned structure to extract protein atoms
                atoms = parsePDB(str(temp_aligned))
                receptor_atoms = atoms.select("protein and not water and not hetero")
                
                # Write protein atoms to a temporary file
                writePDB(str(temp_protein), receptor_atoms)
                
                # Ensure output directory exists
                output_pdb.parent.mkdir(parents=True, exist_ok=True)
                
                # Step 3: Run mk_prepare_receptor.py without box parameters
                result = subprocess.run(
                    [
                        "mk_prepare_receptor.py",
                        "-i", str(temp_protein),
                        "-o", str(output_pdb.with_suffix('')),  # Remove suffix to use as base name
                        "-p",
                        "--allow_bad_res",
                    ],
                    capture_output=True,
                    text=True
                )
                
                self.stdout_meeko = result.stdout
                self.stderr_meeko = result.stderr
                self.success_meeko = result.returncode == 0
                
                # Clean up temporary files
                self._cleanup_temp_files([temp_hydrogenated, temp_aligned, temp_protein])
                
                # Return the path to the prepared protein file
                pdbqt_path = output_pdb.with_suffix('.pdbqt')
                return pdbqt_path if pdbqt_path.exists() else None
        except Exception as e:
            # Capture any errors from PyMOL or ProDy
            if 'prody' in str(e).lower():
                self.prody_error = str(e)
            elif 'pymol' in str(e).lower():
                self.pymol_error = str(e)
            else:
                # Generic error handling
                if crystal_ligand_sdf:
                    return {
                        "pdbqt_path": None,
                        "box_pdb": None,
                        "box_txt": None
                    }
                else:
                    return None
            
            # Clean up temporary files
            self._cleanup_temp_files([temp_hydrogenated, temp_aligned, temp_protein])
            
            # Return appropriate structure based on whether crystal_ligand_sdf was provided
            if crystal_ligand_sdf:
                return {
                    "pdbqt_path": None,
                    "box_pdb": None,
                    "box_txt": None
                }
            else:
                return None
    
    def _cleanup_temp_files(self, file_list):
        """Helper method to clean up temporary files."""
        for file_path in file_list:
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup
    
    def __call__(self, input_pdb: Path, output_pdb: Path, crystal_ligand_sdf: Path = None) -> dict:
        """
        Call method that wraps prepare_protein_meeko for easier usage.
        
        Args:
            input_pdb (Path): Path to the input protein file in PDB format.
            output_pdb (Path): Path to save the prepared protein file.
            crystal_ligand_sdf (Path, optional): Path to the crystal ligand in SDF format.
                                               If provided, box dimensions will be calculated.
        
        Returns:
            dict or Path: If crystal_ligand_sdf is provided, returns a dictionary with paths and contents.
                         Otherwise, returns the path to the prepared protein file in PDBQT format.
        """
        return self.prepare_protein_meeko(input_pdb, output_pdb, crystal_ligand_sdf)


class ProteinPreparation_Chimera:
    """
    A class for preparing protein structures using UCSF Chimera.
    """
    
    def __init__(self):
        """
        Initialize the ProteinPreparation_Chimera class.
        """
        pass
        
    def _set_chimera_env_variable(self):
        """
        Set the environment variable for UCSF Chimera.
        """
        os.environ["PATH"] = "/home/hitesit/.local/UCSF-Chimera64-1.16/bin/:" + os.environ.get("PATH", "")
    
    def _write_chimera_script(self):
        """
        Write a Chimera script for protein preparation.
        
        Returns:
            Path: Path to the Chimera script.
        """
        dockprep_str = f"""
import chimera
import sys
from DockPrep import prep
models = chimera.openModels.list(modelTypes=[chimera.Molecule])
prep(models)
from WriteMol2 import writeMol2
writeMol2(models, "rec_prep.mol2")
        """

        chimera_py_path = os.path.join(tempfile.gettempdir(), "chimera.py")
        with open(chimera_py_path, "w") as f:
            f.write(dockprep_str)

        return Path(chimera_py_path)
    
    def prepare_protein_chimera(self, input_pdb: Path, output_pdb: Path) -> Path:
        """
        Prepares a protein using UCSF Chimera.
        
        Args:
            input_pdb (Path): Path to the input protein file in PDB format.
            output_pdb (Path): Path to save the prepared protein file in MOL2 format.
            
        Returns:
            Path: Path to the prepared protein file in MOL2 format.
        """
        # Print log message
        print('Preparing protein with Chimera ...')
        
        # Set the environment variable
        self._set_chimera_env_variable()
        
        # Write the Chimera script
        chimera_py_path = self._write_chimera_script()
        
        # Execute the script
        chimera_command = f"chimera --nogui {str(input_pdb)} {chimera_py_path}"
        subprocess.run(chimera_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Handle output
        output_system = Path(os.path.join(tempfile.gettempdir(), "rec_prep.mol2"))
        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(output_system, output_pdb)
        
        return output_pdb
    
    def __call__(self, input_pdb: Path, output_pdb: Path) -> Path:
        """
        Call method that wraps prepare_protein_chimera for easier usage.
        
        Args:
            input_pdb (Path): Path to the input protein file in PDB format.
            output_pdb (Path): Path to save the prepared protein file in MOL2 format.
            
        Returns:
            Path: Path to the prepared protein file in MOL2 format.
        """
        return self.prepare_protein_chimera(input_pdb, output_pdb)
