"""
Robust solution for fixing mol2 docking poses with no bond order information.
Handles the most challenging cases where standard methods fail.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import datamol as dm
from typing import List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_docking_poses_from_mol2(
    broken_mols: List[Chem.Mol],
    template_smiles: str,
    verbose: bool = True
) -> List[Optional[Chem.Mol]]:
    """
    Main function to fix bond orders in mol2 docking poses.
    
    This handles the extreme case where docking engines output molecules
    in a completely unkekulizable form with no bond order information.
    
    Args:
        broken_mols: List from dm.read_mol2file(file, sanitize=False)
        template_smiles: Kekulized SMILES of the correct molecule
        verbose: Print progress information
    
    Returns:
        List of fixed molecules with correct bond orders and preserved 3D coordinates
    """
    # Prepare template
    template = Chem.MolFromSmiles(template_smiles)
    if not template:
        raise ValueError(f"Invalid template SMILES: {template_smiles}")
    
    # Remove hydrogens from template (since mol2 has no H)
    template = Chem.RemoveHs(template)
    
    # Process each pose
    fixed_molecules = []
    for i, mol in enumerate(broken_mols):
        if verbose:
            print(f"\nProcessing pose {i+1}/{len(broken_mols)}")
        
        fixed = fix_single_pose(mol, template, verbose)
        fixed_molecules.append(fixed)
        
        if verbose:
            status = "✓ Success" if fixed else "✗ Failed"
            print(f"  {status}")
    
    # Summary
    if verbose:
        n_fixed = sum(1 for m in fixed_molecules if m is not None)
        print(f"\n{'='*50}")
        print(f"Total: {n_fixed}/{len(broken_mols)} poses fixed successfully")
        print(f"{'='*50}")
    
    return fixed_molecules


def fix_single_pose(
    broken_mol: Chem.Mol,
    template: Chem.Mol,
    verbose: bool = True
) -> Optional[Chem.Mol]:
    """
    Fix a single broken molecule using a cascade of strategies.
    """
    strategies = [
        ("Template reconstruction", strategy_template_reconstruction),
        ("Force bond assignment", strategy_force_bond_assignment),
        ("Connectivity matching", strategy_connectivity_matching),
    ]
    
    for name, strategy in strategies:
        if verbose:
            print(f"  Trying: {name}...", end=" ")
        
        try:
            result = strategy(broken_mol, template)
            if result is not None:
                if verbose:
                    print("Success!")
                return result
            else:
                if verbose:
                    print("Failed")
        except Exception as e:
            if verbose:
                print(f"Error: {str(e)[:50]}...")
    
    return None


def strategy_template_reconstruction(
    broken_mol: Chem.Mol,
    template: Chem.Mol
) -> Optional[Chem.Mol]:
    """
    Strategy 1: Create new molecule from template and transfer coordinates.
    This is the most reliable method when atoms can be matched.
    """
    # Prepare broken molecule for matching
    working_mol = copy.deepcopy(broken_mol)
    
    # Reset all bonds to single and clear aromaticity
    for bond in working_mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in working_mol.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)  # Reset charges
    
    working_mol.UpdatePropertyCache(strict=False)
    
    # Try to find atom mapping
    matches = working_mol.GetSubstructMatches(template, useChirality=False)
    if not matches:
        # Try with more relaxed matching
        working_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(working_mol, 
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                        catchErrors=True)
        matches = working_mol.GetSubstructMatches(template, useChirality=False)
    
    if not matches:
        return None
    
    # Use first match
    atom_map = matches[0]
    
    # Create new molecule from template
    new_mol = Chem.RWMol(template)
    
    # Transfer coordinates
    if broken_mol.GetNumConformers() > 0:
        conf = Chem.Conformer(new_mol.GetNumAtoms())
        broken_conf = broken_mol.GetConformer()
        
        for template_idx in range(template.GetNumAtoms()):
            broken_idx = atom_map[template_idx]
            pos = broken_conf.GetAtomPosition(broken_idx)
            conf.SetAtomPosition(template_idx, pos)
        
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf)
    
    # Finalize
    final_mol = new_mol.GetMol()
    
    # Sanitize and assign stereochemistry
    try:
        Chem.SanitizeMol(final_mol)
        if final_mol.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(final_mol)
        return final_mol
    except:
        return None


def strategy_force_bond_assignment(
    broken_mol: Chem.Mol,
    template: Chem.Mol
) -> Optional[Chem.Mol]:
    """
    Strategy 2: Force bond order assignment with multiple attempts.
    """
    # Make working copy
    working_mol = copy.deepcopy(broken_mol)
    
    # Clean up the molecule as much as possible
    for bond in working_mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    
    for atom in working_mol.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
    
    # Update properties
    working_mol.UpdatePropertyCache(strict=False)
    
    # Try standard assignment with error catching
    try:
        result = AllChem.AssignBondOrdersFromTemplate(template, working_mol)
        # Verify it worked
        Chem.SanitizeMol(result)
        if result.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(result)
        return result
    except:
        pass
    
    # Try with partial sanitization first
    try:
        flags = (Chem.SanitizeFlags.SANITIZE_SYMMRINGS | 
                Chem.SanitizeFlags.SANITIZE_CLEANUP)
        Chem.SanitizeMol(working_mol, sanitizeOps=flags, catchErrors=True)
        working_mol.UpdatePropertyCache(strict=False)
        
        result = AllChem.AssignBondOrdersFromTemplate(template, working_mol)
        Chem.SanitizeMol(result)
        if result.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(result)
        return result
    except:
        return None


def strategy_connectivity_matching(
    broken_mol: Chem.Mol,
    template: Chem.Mol
) -> Optional[Chem.Mol]:
    """
    Strategy 3: Match by connectivity patterns when substructure matching fails.
    """
    if broken_mol.GetNumAtoms() != template.GetNumAtoms():
        return None
    
    # Build connectivity signatures
    def get_connectivity_signature(mol):
        sigs = []
        for atom in mol.GetAtoms():
            neighbors = sorted([n.GetSymbol() for n in atom.GetNeighbors()])
            sig = (atom.GetSymbol(), atom.GetDegree(), tuple(neighbors))
            sigs.append(sig)
        return sigs
    
    broken_sigs = get_connectivity_signature(broken_mol)
    template_sigs = get_connectivity_signature(template)
    
    # Try to find mapping
    mapping = find_best_atom_mapping(broken_sigs, template_sigs)
    if not mapping:
        return None
    
    # Build new molecule using mapping
    new_mol = Chem.RWMol()
    
    # Add atoms
    for i in range(template.GetNumAtoms()):
        atom = Chem.Atom(template.GetAtomWithIdx(i).GetAtomicNum())
        new_mol.AddAtom(atom)
    
    # Add bonds from template
    for bond in template.GetBonds():
        new_mol.AddBond(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond.GetBondType()
        )
    
    # Copy coordinates using mapping
    if broken_mol.GetNumConformers() > 0:
        conf = Chem.Conformer(new_mol.GetNumAtoms())
        broken_conf = broken_mol.GetConformer()
        
        for template_idx, broken_idx in enumerate(mapping):
            if broken_idx is not None:
                pos = broken_conf.GetAtomPosition(broken_idx)
                conf.SetAtomPosition(template_idx, pos)
        
        new_mol.AddConformer(conf)
    
    # Finalize
    final_mol = new_mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol)
        if final_mol.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(final_mol)
        return final_mol
    except:
        return None


def find_best_atom_mapping(sigs1: List, sigs2: List) -> Optional[List[int]]:
    """
    Find best mapping between two sets of connectivity signatures.
    """
    n = len(sigs1)
    if n != len(sigs2):
        return None
    
    mapping = [None] * n
    used = set()
    
    # First pass: exact matches
    for i, sig2 in enumerate(sigs2):
        exact_matches = [j for j, sig1 in enumerate(sigs1) 
                        if sig1 == sig2 and j not in used]
        if len(exact_matches) == 1:
            mapping[i] = exact_matches[0]
            used.add(exact_matches[0])
    
    # Second pass: match by element only
    for i, sig2 in enumerate(sigs2):
        if mapping[i] is None:
            element = sig2[0]
            candidates = [j for j, sig1 in enumerate(sigs1)
                         if sig1[0] == element and j not in used]
            if candidates:
                mapping[i] = candidates[0]
                used.add(candidates[0])
    
    # Check completeness
    if None in mapping:
        return None
    
    return mapping


# Convenience function for direct file processing
def hard_fix_mol2(
    mol2_filename: str,
    template_smiles: str,
    output_sdf: Optional[str] = None,
    reader_func=dm.read_mol2file
) -> None:
    """
    Process a mol2 file and write fixed molecules to SDF.
    
    Args:
        mol2_filename: Input mol2 file
        template_smiles: Kekulized SMILES
        output_sdf: Output SDF file
        reader_func: Function to read mol2 (e.g., dm.read_mol2file)
    """
    # Read molecules
    if reader_func:
        mols = reader_func(mol2_filename, sanitize=False)
    else:
        # Fallback to RDKit's reader
        mols = []
        with open(mol2_filename, 'r') as f:
            mol_block = []
            for line in f:
                if line.strip() == '@<TRIPOS>MOLECULE':
                    if mol_block:
                        mol_text = ''.join(mol_block)
                        mol = Chem.MolFromMol2Block(mol_text, sanitize=False)
                        if mol:
                            mols.append(mol)
                    mol_block = [line]
                else:
                    mol_block.append(line)
            # Don't forget last molecule
            if mol_block:
                mol_text = ''.join(mol_block)
                mol = Chem.MolFromMol2Block(mol_text, sanitize=False)
                if mol:
                    mols.append(mol)
    
    print(f"Read {len(mols)} molecules from {mol2_filename}")
    
    # Fix molecules
    fixed_mols = fix_docking_poses_from_mol2(mols, template_smiles)
    
    # Write results
    if output_sdf:
        writer = Chem.SDWriter(output_sdf)
        n_written = 0
        for i, mol in enumerate(fixed_mols):
            if mol is not None:
                mol.SetProp("_Name", f"Pose_{i+1}")
                mol.SetProp("OriginalPoseNumber", str(i+1))
                writer.write(mol)
                n_written += 1
        writer.close()
        
        print(f"\nWrote {n_written} molecules to {output_sdf}")
    else:
        return fixed_mols