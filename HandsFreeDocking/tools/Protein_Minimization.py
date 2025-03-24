from sys import stdout
import numpy as np
import io
import copy
import cloudpickle
from tempfile import NamedTemporaryFile
from pdbfixer import PDBFixer
from typing import List, Dict, Any, Optional, Union, Tuple, TypeVar, IO, BinaryIO, TextIO
from pathlib import Path
import os
import tempfile

# ParmEd
import parmed as pmd
import pytraj as pt

# Biotite
import biotite.structure.io.pdb as pdb
from biotite.structure import AtomArray

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
import datamol as dm

# OpenMM Application Layer
from openmm import app
from openmm import XmlSerializer
from openmm.app import (
    Modeller, Simulation, PDBFile, DCDReporter, 
    StateDataReporter, CheckpointReporter
)

# OpenMM Library Layer
from openmm import (
    Platform, LangevinIntegrator, MonteCarloBarostat,
    CustomExternalForce, State, System, Context
)

# OpenMM Units
from openmm import unit
from openmm.unit import Quantity

# OPENFF
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator

# Local
from HandsFreeDocking.analysis.clustering import calc_rmsd_mcs_with_timeout

# Type for OpenMM unit quantities
UnitQuantity = Quantity

# def extract_unk_residue(modeller: Modeller) -> Tuple[app.Topology, unit.Quantity]:
#     """
#     Extract only the UNK residue from a modeller object without modifying the original.
#     
#     Parameters
#     ----------
#     modeller : openmm.app.Modeller
#         The modeller object containing the full topology
#         
#     Returns
#     -------
#     topology : openmm.app.Topology
#         Topology containing only the UNK residue
#     positions : openmm.unit.Quantity
#         Positions of the atoms in the UNK residue with appropriate units
#     """
#     # Create a copy of the modeller to preserve the original
#     unk_modeller: Modeller = copy.deepcopy(modeller)
#     
#     # Find the UNK residue in the copy
#     unk_residues: List[app.Residue] = [res for res in unk_modeller.topology.residues() if res.name == "UNK"]
#     if not unk_residues:
#         raise ValueError("No residue named 'UNK' found")
#         
#     unk_residue: app.Residue = unk_residues[0]
#     
#     # Get a list of all atoms that are NOT in the UNK residue
#     atoms_to_delete: List[app.Atom] = []
#     for atom in unk_modeller.topology.atoms():
#         if atom.residue != unk_residue:
#             atoms_to_delete.append(atom)
#     
#     # Delete all atoms not in the UNK residue
#     unk_modeller.delete(atoms_to_delete)
#     
#     # Return the topology and positions
#     return unk_modeller.topology, unk_modeller.positions

def minimize_complex(prot_path: Union[str, Path], lig_mol: Chem.rdchem.Mol) -> Dict[str, Union[str, float]]:
    """
    Prepare and minimize a protein-ligand complex using OpenMM.
    
    This function:
    1. Processes the protein structure to fix missing atoms and add hydrogens
    2. Combines the protein with the provided ligand molecule
    3. Sets up a simulation system with appropriate forcefields
    4. Performs energy minimization on the complex
    5. Returns the minimized structure as a PDB string
    
    Parameters
    ----------
    prot_path : Union[str, Path]
        Path to the protein PDB file
    lig_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object representing the ligand
        
    Returns
    -------
    Dict[str, Union[str, float]]
        Dictionary containing:
        - "PDB_BEFORE": PDB string of the initial complex
        - "PDB_AFTER": PDB string of the minimized complex
        - "energy_before_min": Energy before minimization (kJ/mol)
        - "energy_after_min": Energy after minimization (kJ/mol)
    """
    # Fix the protein
    fixer: PDBFixer = PDBFixer(str(prot_path))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    
    # Parse the ligand
    ligand_mol: Molecule = Molecule.from_rdkit(lig_mol)
    lig_top = ligand_mol.to_topology()
    
    # Merge the ligand into the protein
    modeller: Modeller = Modeller(fixer.topology, fixer.positions)
    modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
    
    # Create the forcefield
    forcefield_kwargs: Dict[str, Any] = { 
        'constraints': app.HBonds, 
        # 'rigidWater': True, 
        # 'removeCMMotion': False, 
        'hydrogenMass': 4*unit.amu 
    }
    
    system_generator: SystemGenerator = SystemGenerator(
        forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=[ligand_mol],
        forcefield_kwargs=forcefield_kwargs
    )
    
    system: System = system_generator.create_system(modeller.topology)
    integrator: LangevinIntegrator = LangevinIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    
    platform: Platform = Platform.getPlatformByName('CUDA')
    proprieties: Dict[str, str] = {'Precision': 'mixed', 'CudaDeviceIndex': "0"}

    simulation: Simulation = Simulation(
        modeller.topology, 
        system,
        integrator, 
        platform=platform, 
        platformProperties=proprieties
    )
    
    # Set context
    context: Context = simulation.context
    context.setPositions(modeller.positions)
    
    # Minimize
    min_state: State = simulation.context.getState(getEnergy=True, getPositions=True)
    energy_before_min: UnitQuantity = min_state.getPotentialEnergy()
    simulation.minimizeEnergy()
    min_state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy_after_min: UnitQuantity = min_state.getPotentialEnergy()
    
    # with io.StringIO() as pdb_string:
    #     PDBFile.writeHeader(simulation.topology, pdb_string)
    #     PDBFile.writeModel(modeller.topology, modeller.positions, pdb_string, modelIndex=1)
    #     PDBFile.writeModel(simulation.topology, min_state.getPositions(), pdb_string, modelIndex=2)
    #     PDBFile.writeFooter(simulation.topology, pdb_string)
    #     PDB_MIN = pdb_string.getvalue()
        
    with io.StringIO() as PDB_before:
        PDBFile.writeFile(modeller.topology, modeller.positions, PDB_before)
        PDB_BEFORE: str = PDB_before.getvalue()
    
    with io.StringIO() as PDB_after:
        PDBFile.writeFile(simulation.topology, min_state.getPositions(), PDB_after)
        PDB_AFTER: str = PDB_after.getvalue()
    
    # Get the energies
    energy_before_min: float = energy_before_min.value_in_unit(unit.kilojoule_per_mole)
    energy_after_min: float = energy_after_min.value_in_unit(unit.kilojoule_per_mole)
    
    delta_energy = energy_after_min - energy_before_min
    
    return {
        "PDB_BEFORE": PDB_BEFORE,
        "PDB_AFTER": PDB_AFTER,
        "delta_energy": round(delta_energy, 2)
    }

def assign_bond_orders_from_smiles(
    pdb_input: Union[str, Path, io.StringIO], 
    smiles: str, 
    output_sdf: Optional[Union[str, Path]] = None
) -> Optional[Chem.rdchem.Mol]:
    """
    Assign bond orders to a 3D molecule using a SMILES string.
    
    Parameters:
    -----------
    pdb_input : Union[str, Path, io.StringIO]
        The PDB file as a path, PDB content as a string, or StringIO object
    smiles : str
        SMILES string of the ligand
    output_sdf : Optional[Union[str, Path]]
        Path to save the output SDF file. If None, will return the molecule object
    
    Returns:
    --------
    Optional[rdkit.Chem.rdchem.Mol]
        RDKit molecule with assigned bond orders if output_sdf is None, otherwise None
    """
    # Handle different input types for PDB
    temp_file: Optional[tempfile.NamedTemporaryFile] = None
    
    try:
        if isinstance(pdb_input, (str, Path)):
            # Check if it's a file path or PDB content string
            if os.path.exists(str(pdb_input)):
                # It's a file path
                pdb_file: str = str(pdb_input)
            else:
                # It's a PDB content string
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='w')
                temp_file.write(pdb_input)
                temp_file.close()
                pdb_file = temp_file.name
        
        elif isinstance(pdb_input, io.StringIO):
            # It's a StringIO object
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='w')
            temp_file.write(pdb_input.getvalue())
            temp_file.close()
            pdb_file = temp_file.name
            
        else:
            raise ValueError("pdb_input must be a string, Path, or StringIO object")
        
        # Read the PDB file
        mol_3d: Chem.rdchem.Mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        if mol_3d is None:
            raise ValueError(f"Could not read PDB content")
        
        # Create molecule from SMILES
        mol_2d: Chem.rdchem.Mol = Chem.MolFromSmiles(smiles)
        if mol_2d is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Get 3D coordinates from the PDB molecule
        conf: Chem.rdchem.Conformer = mol_3d.GetConformer()
        coords: np.ndarray = np.array([conf.GetAtomPosition(i) for i in range(mol_3d.GetNumAtoms())])
        
        # Prepare the template molecule from SMILES with hydrogen atoms
        mol_template: Chem.rdchem.Mol = Chem.AddHs(mol_2d)
        
        # Match atoms between the molecules
        num_atoms_3d: int = mol_3d.GetNumAtoms()
        num_atoms_template: int = mol_template.GetNumAtoms()
        
        if num_atoms_3d != num_atoms_template:
            print(f"Warning: Different number of atoms. PDB: {num_atoms_3d}, SMILES: {num_atoms_template}")
        
        # Create a new molecule with correct bond orders
        mol_result: Chem.rdchem.Mol = AllChem.AssignBondOrdersFromTemplate(mol_template, mol_3d)
        
        # If successful, save or return
        if mol_result is None:
            raise ValueError("Failed to assign bond orders. Molecules may not be compatible.")
        
        if output_sdf:
            # Write to SDF file
            writer: Chem.SDWriter = Chem.SDWriter(str(output_sdf))
            writer.write(mol_result)
            writer.close()
            print(f"Successfully saved molecule with bond orders to {output_sdf}")
            return None
        else:
            return mol_result
            
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def extract_unk_residue(pdb_str: str) -> Path:
    """
    Extract the UNK residue from a PDB string.
    
    Parameters
    ----------
    pdb_str : str
        PDB string containing the structure
        
    Returns
    -------
    biotite.structure.AtomArray
        Atom array containing only the UNK residue
    """
    pbd_io: io.StringIO = io.StringIO(pdb_str)
    pdb_struct: AtomArray = pdb.PDBFile.read(pbd_io).get_structure(model=1)
    unk_struct: AtomArray = pdb_struct[pdb_struct.res_name == "UNK"]
    
    unk_file: pdb.PDBFile = pdb.PDBFile()
    unk_file.set_structure(unk_struct)
    
    with NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_file:
        unk_file.write(tmp_file.name)
        return Path(tmp_file.name)

class ProteinMinimizer:
    def __init__(self, df, docked_mol_col: str, pdb_path_col: str):
        """
        Initialize the protein minimizer.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing molecules and paths
        docked_mol_col : str
            Column name containing RDKit molecules
        pdb_path_col : str
            Column name containing paths to PDB files
        """
        self.df = df.copy()
        self.docked_mol_col = docked_mol_col
        self.pdb_path_col = pdb_path_col
        
    def _apply_on_row(self, row):
        """
        Process a single row from the DataFrame.
        
        Parameters
        ----------
        row : pandas.Series
            A row from the DataFrame
            
        Returns
        -------
        dict
            Dictionary containing minimization results
        """
        try:
            # Extract the docked molecule and PDB path
            mol_docked = row[self.docked_mol_col]
            if mol_docked is None:
                return {'after_min': None, 'rmsd_val': None, 'delta_energy': None, 'error': 'No molecule found'}
                
            # Convert molecule to SMILES
            smiles_docked = dm.to_smiles(mol_docked, kekulize=True)
            if not smiles_docked:
                return {'after_min': None, 'rmsd_val': None, 'delta_energy': None, 'error': 'Failed to generate SMILES'}
                
            # Get PDB path
            pdb_path = Path(row[self.pdb_path_col])
            if not pdb_path.exists():
                return {'after_min': None, 'rmsd_val': None, 'delta_energy': None, 'error': f'PDB file not found: {pdb_path}'}
            
            # Perform minimization
            minimization_dict = minimize_complex(pdb_path, mol_docked)
            after_min = minimization_dict["PDB_AFTER"]
            delta_energy = minimization_dict["delta_energy"]
            
            # Extract UNK residue and assign bond orders
            after_unk = extract_unk_residue(after_min)
            after_mol = assign_bond_orders_from_smiles(after_unk, smiles_docked)
            
            # Calculate RMSD
            rmsd_val = calc_rmsd_mcs_with_timeout(mol_docked, after_mol)
            
            # Clean up temporary file
            if after_unk.exists():
                try:
                    after_unk.unlink()
                except:
                    pass
                
            return {
                'after_min': after_min,
                'rmsd_val': rmsd_val,
                'delta_energy': delta_energy,
                'error': None
            }
            
        except Exception as e:
            import traceback
            error_message = f"Error processing row: {str(e)}\n{traceback.format_exc()}"
            print(error_message)
            return {
                'after_min': None,
                'rmsd_val': None,
                'delta_energy': None,
                'error': str(e)
            }
    
    def __call__(self):
        """
        Process all rows in the DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with added minimization results
        """
        from tqdm import tqdm
        
        # Create empty columns to store results
        self.df['PDB_Min'] = None
        self.df['Delta_RMSD'] = None
        self.df['Delta_Energy'] = None
        self.df['ERROR'] = None
        
        # Process each row
        tqdm.pandas(desc="Minimizing proteins")
        
        # Apply the function and create a list of dictionaries
        results = self.df.progress_apply(self._apply_on_row, axis=1)
        
        # Update the DataFrame with the results
        for idx, result in zip(self.df.index, results):
            if result:
                self.df.at[idx, 'PDB_Min'] = result.get('after_min')
                self.df.at[idx, 'Delta_RMSD'] = result.get('rmsd_val')
                self.df.at[idx, 'Delta_Energy'] = result.get('delta_energy')
                self.df.at[idx, 'ERROR'] = result.get('error')
        
        return self.df