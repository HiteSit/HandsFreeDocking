"""
Unified Ligand Preparation Module

This module provides a unified interface for ligand preparation across different
docking pipelines. It supports protonation, stereoisomer enumeration, tautomer
enumeration, and 3D conformation generation using either CDPKit or OpenEye toolkits.

The module standardizes ligand preparation across all docking pipelines to ensure
consistency and maintainability.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Literal
import logging
import tempfile

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import rdCIPLabeler

# Datamol for utility functions
import datamol as dm

# OpenBabel for format conversions
from openbabel import pybel

# CDPK imports
import CDPL.Chem as CDPLChem
import CDPL.ConfGen as ConfGen

# Optional OpenEye imports
try:
    from openeye import oechem
    from openeye import oequacpac
    from openeye import oeomega
    OPENEYE_AVAILABLE = True
except ImportError:
    OPENEYE_AVAILABLE = False

# Optional scrubber imports for protonation
try:
    from scrubber import Scrub
    SCRUBBER_AVAILABLE = True
except ImportError:
    SCRUBBER_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LigandPreparator:
    """
    Unified ligand preparation class that handles protonation, stereoisomer enumeration,
    tautomer enumeration, and 3D conformation generation.
    
    This class provides a consistent interface for ligand preparation across different
    docking pipelines, supporting both CDPKit and OpenEye toolkits.
    """
    
    def __init__(self, 
                 protonation_method: str = "cdp",
                 enumerate_stereo: bool = True,
                 tautomer_score_threshold: Optional[float] = None,
                 generate_3d: bool = True):
        """
        Initialize the LigandPreparator.
        
        Args:
            protonation_method: Method for protonation ("cdp", "oe", or "scrubber")
            enumerate_stereo: Whether to enumerate stereoisomers
            tautomer_score_threshold: Score threshold for tautomer selection (None = best only, value = list within threshold)
            generate_3d: Whether to generate 3D conformations
        """
        self.protonation_method = protonation_method.lower()
        self.enumerate_stereo = enumerate_stereo
        self.tautomer_score_threshold = tautomer_score_threshold
        self.generate_3d = generate_3d
        
        # Validate protonation method
        if self.protonation_method not in ["cdp", "oe", "scrubber"]:
            raise ValueError(f"Invalid protonation method: {self.protonation_method}. Must be 'cdp', 'oe', or 'scrubber'")
        
        if self.protonation_method == "oe" and not OPENEYE_AVAILABLE:
            logger.warning("OpenEye not available for protonation! Falling back to CDP.")
            self.protonation_method = "cdp"
            
        if self.protonation_method == "scrubber" and not SCRUBBER_AVAILABLE:
            logger.warning("Scrubber not available for protonation! Falling back to CDP.")
            self.protonation_method = "cdp"
    
    def prepare_from_smiles(self, smiles_list: List[str], names: Optional[List[str]] = None) -> List[Chem.Mol]:
        """
        Prepare ligands from a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            names: Optional list of molecule names (must match length of smiles_list)
            
        Returns:
            List of prepared RDKit molecules
        """
        if names is not None and len(names) != len(smiles_list):
            raise ValueError("Length of names must match length of smiles_list")
        
        prepared_molecules = []
        
        for i, smiles in enumerate(smiles_list):
            mol_name = names[i] if names else f"mol_{i}"
            logger.info(f"Processing {mol_name} from SMILES: {smiles}")
            
            try:
                # Process single SMILES through the pipeline
                processed_mols = self._process_single_molecule(smiles, mol_name)
                prepared_molecules.extend(processed_mols)
            except Exception as e:
                logger.error(f"Failed to process {mol_name}: {e}")
                continue
        
        return prepared_molecules
    
    def prepare_from_sdf(self, sdf_path: Path) -> List[Chem.Mol]:
        """
        Prepare ligands from an SDF file.
        
        Args:
            sdf_path: Path to the input SDF file
            
        Returns:
            List of prepared RDKit molecules
        """
        prepared_molecules = []
        
        # Read molecules from SDF
        supplier = Chem.SDMolSupplier(str(sdf_path))
        
        for i, mol in enumerate(supplier):
            if mol is None:
                logger.warning(f"Failed to read molecule {i} from {sdf_path}")
                continue
            
            mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
            smiles = Chem.MolToSmiles(mol)
            logger.info(f"Processing {mol_name} from SDF")
            
            try:
                # Process single molecule through the pipeline
                processed_mols = self._process_single_molecule(smiles, mol_name)
                prepared_molecules.extend(processed_mols)
            except Exception as e:
                logger.error(f"Failed to process {mol_name}: {e}")
                continue
        
        return prepared_molecules
    
    def _process_single_molecule(self, smiles: str, base_name: str) -> List[Chem.Mol]:
        """
        Process a single molecule through the preparation pipeline.
        
        Args:
            smiles: SMILES string of the molecule
            base_name: Base name for the molecule
            
        Returns:
            List of prepared RDKit molecules
        """
        # Step 1: Protonation
        protonated_smiles = self._protonate_molecule(smiles)
        
        # Step 2: Stereoisomer enumeration
        if self.enumerate_stereo:
            stereo_mols = self._enumerate_stereoisomers(protonated_smiles, base_name)
        else:
            mol = Chem.MolFromSmiles(protonated_smiles)
            mol.SetProp("_Name", base_name)
            stereo_mols = [mol]
        
        # Step 3: Tautomer enumeration 
        final_molecules = []
        for stereo_mol in stereo_mols:
            stereo_name = stereo_mol.GetProp("_Name")
            tautomer_result = self._get_best_tautomer(stereo_mol)
            
            if isinstance(tautomer_result, list):
                # Multiple tautomers returned (when tautomer_score_threshold is provided)
                for j, taut in enumerate(tautomer_result):
                    taut_name = f"{stereo_name}_Taut{j}"
                    taut.SetProp("_Name", taut_name)
                    final_molecules.append(taut)
            else:
                # Single tautomer returned (when tautomer_score_threshold is None)
                if tautomer_result is not None:
                    tautomer_result.SetProp("_Name", stereo_name)
                    final_molecules.append(tautomer_result)
                else:
                    final_molecules.append(stereo_mol)
        
        # Step 4: 3D conformation generation
        if self.generate_3d:
            molecules_3d = []
            for mol in final_molecules:
                mol_3d = self._generate_3d_conformation(mol)
                if mol_3d is not None:
                    molecules_3d.append(mol_3d)
            return molecules_3d
        
        return final_molecules
    
    def _protonate_molecule(self, smiles: str) -> str:
        """
        Protonate a molecule at physiological pH.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Protonated SMILES string
        """
        if self.protonation_method == "oe":
            return self._oe_protonate(smiles)
        elif self.protonation_method == "scrubber":
            return self._scrubber_protonate(smiles)
        else:  # cdp
            return self._cdp_protonate(smiles)
    
    def _cdp_protonate(self, smiles: str) -> str:
        """Protonate using CDPKit."""
        # Parse the SMILES string to create a molecule object
        mol = CDPLChem.parseSMILES(smiles)
        
        if mol is None:
            raise Exception(f"Failed to parse SMILES: {smiles}")
        
        # Create a BasicMolecule to store the result
        out_mol = CDPLChem.BasicMolecule(mol)
        
        # Prepare the molecule - calculate required properties
        CDPLChem.calcImplicitHydrogenCounts(out_mol, False)
        CDPLChem.perceiveHybridizationStates(out_mol, False)
        CDPLChem.perceiveSSSR(out_mol, False)
        CDPLChem.setRingFlags(out_mol, False)
        CDPLChem.setAromaticityFlags(out_mol, False)
        CDPLChem.perceiveComponents(out_mol, False)
        
        # Create and apply the protonation state standardizer
        prot_state_gen = CDPLChem.ProtonationStateStandardizer()
        
        # Apply physiological condition protonation
        prot_state_gen.standardize(out_mol, CDPLChem.ProtonationStateStandardizer.PHYSIOLOGICAL_CONDITION_STATE)
        
        # Update component perception as structure might have changed
        CDPLChem.perceiveComponents(out_mol, True)
        
        # Generate and return the SMILES string
        return CDPLChem.generateSMILES(out_mol, False, True)
    
    def _oe_protonate(self, smiles: str) -> str:
        """Protonate using OpenEye."""
        # Convert SMILES to OpenEye molecule
        oemol = oechem.OEMol()
        oechem.OESmilesToMol(oemol, smiles)

        # Strip Salt
        oechem.OEDeleteEverythingExceptTheFirstLargestComponent(oemol)

        oechem.OEFindRingAtomsAndBonds(oemol)
        oechem.OEAssignAromaticFlags(oemol, oechem.OEAroModel_OpenEye)
        
        oechem.OEAddExplicitHydrogens(oemol)
        oequacpac.OEGetReasonableProtomer(oemol)

        smiflag = oechem.OESMILESFlag_Canonical | oechem.OESMILESFlag_ISOMERIC
        return oechem.OECreateSmiString(oemol, smiflag)
    
    def _scrubber_protonate(self, smiles: str) -> str:
        """Protonate using Scrubber."""
        mol = dm.to_mol(smiles)

        scrub = Scrub(
            ph_low=7.4,
            ph_high=7.4,
        )

        with dm.without_rdkit_log():
            prot_mols = list(scrub(mol))
            if len(prot_mols) > 0:
                best_mol = prot_mols[0]
                best_mol_noH = dm.remove_hs(best_mol)

        return dm.to_smiles(best_mol_noH, explicit_hs=False)
    
    def _enumerate_stereoisomers(self, smiles: str, base_name: str) -> List[Chem.Mol]:
        """
        Enumerate stereoisomers for a molecule.
        
        Args:
            smiles: SMILES string
            base_name: Base name for naming molecules
            
        Returns:
            List of RDKit molecules with stereoisomer information
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to create molecule from SMILES: {smiles}")
            return []
        
        opts = StereoEnumerationOptions(tryEmbedding=False, unique=True, onlyUnassigned=True)
        
        # Enumerate stereoisomers
        isomers_tuple = EnumerateStereoisomers(mol, options=opts)
        isomers = list(isomers_tuple)
        
        # If no stereoisomers found, return the original molecule
        if not isomers:
            mol.SetProp("_Name", f"{base_name}_Iso0")
            mol.SetProp("STEREO", "No assignable chiral centers found or molecule is achiral.")
            return [mol]
        
        # Annotate each stereoisomer
        annotated_isomers = []
        for i, isomer in enumerate(isomers):
            # Assign stereochemistry
            Chem.AssignStereochemistry(isomer, cleanIt=True, force=True, flagPossibleStereoCenters=True)
            
            # Find chiral centers
            chiral_centers = Chem.FindMolChiralCenters(isomer, includeUnassigned=True)
            
            # Build stereochemistry info
            stereo_info_parts = []
            if chiral_centers:
                for atom_idx, chirality_tag in chiral_centers:
                    atom = isomer.GetAtomWithIdx(atom_idx)
                    atom_symbol = atom.GetSymbol()
                    stereo_info_parts.append(f"Atom {atom_idx} ({atom_symbol}): {chirality_tag}")
            
            # Set properties
            isomer.SetProp("_Name", f"{base_name}_Iso{i}")
            if stereo_info_parts:
                isomer.SetProp("STEREO", "; ".join(stereo_info_parts))
            else:
                isomer.SetProp("STEREO", "No assignable chiral centers found or molecule is achiral.")
            
            annotated_isomers.append(isomer)
        
        return annotated_isomers
    
    def _get_best_tautomer(self, mol: Chem.Mol) -> Union[Chem.Mol, List[Chem.Mol], None]:
        """
        Get the best tautomer(s) of a molecule based on score threshold.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            If tautomer_score_threshold is None: Single best tautomer (Chem.Mol)
            If tautomer_score_threshold is provided: List of tautomers within threshold (List[Chem.Mol])
            Returns None (single mode) or empty list (list mode) if no tautomers are found.
        """
        params = rdMolStandardize.CleanupParameters()
        params.maxTautomers = 1000
        params.tautomerRemoveSp3Stereo = False
        params.tautomerRemoveBondStereo = False
        params.tautomerRemoveIsotopicHs = False

        enumerator = rdMolStandardize.TautomerEnumerator(params)
        tautomers = list(enumerator.Enumerate(mol))

        if len(tautomers) == 0:
            logger.warning("No tautomers found for the input molecule.")
            return [] if self.tautomer_score_threshold is not None else mol
        
        # Score all tautomers and find the best score
        scored_tautomers = []
        best_score = -float('inf')
        
        for taut in tautomers:
            score = enumerator.ScoreTautomer(taut)
            scored_tautomers.append((taut, score))
            if score > best_score:
                best_score = score
        
        if self.tautomer_score_threshold is None:
            # Return only the single best tautomer
            for taut, score in scored_tautomers:
                if score == best_score:
                    taut.SetProp("TautomerScore", str(score))
                    return taut
        else:
            # Return all tautomers within the threshold as a list
            selected_tautomers = []
            for taut, score in scored_tautomers:
                if abs(score - best_score) <= self.tautomer_score_threshold:
                    taut.SetProp("TautomerScore", str(score))
                    selected_tautomers.append(taut)
            
            return selected_tautomers if selected_tautomers else []
    
    
    def _generate_3d_conformation(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Generate a single 3D conformation for a molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Molecule with 3D coordinates or None if generation fails
        """
        try:
            # Make a copy to avoid modifying the original
            mol_3d = Chem.Mol(mol)
            
            # Add hydrogens
            mol_3d = Chem.AddHs(mol_3d)
            
            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            
            if result != 0:
                logger.warning(f"Failed to generate 3D coordinates for {mol.GetProp('_Name')}")
                return None
            
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
            
            # Remove hydrogens for cleaner output (optional)
            # mol_3d = Chem.RemoveHs(mol_3d)
            
            # Copy properties from original molecule
            for prop_name in mol.GetPropNames():
                mol_3d.SetProp(prop_name, mol.GetProp(prop_name))
            
            return mol_3d
            
        except Exception as e:
            logger.error(f"Error generating 3D conformation: {e}")
            return None
    
    def save_to_sdf(self, molecules: List[Chem.Mol], output_path: Path) -> None:
        """
        Save molecules to an SDF file.
        
        Args:
            molecules: List of RDKit molecules
            output_path: Path for output SDF file
        """
        writer = Chem.SDWriter(str(output_path))
        
        for mol in molecules:
            if mol is not None:
                writer.write(mol)
        
        writer.close()
        logger.info(f"Saved {len(molecules)} molecules to {output_path}")
    
    def save_to_mol2(self, molecules: List[Chem.Mol], output_dir: Path) -> List[Path]:
        """
        Save molecules to individual MOL2 files.
        
        Args:
            molecules: List of RDKit molecules
            output_dir: Directory to save MOL2 files
            
        Returns:
            List of paths to created MOL2 files
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        output_paths = []
        
        for mol in molecules:
            if mol is None:
                continue
            
            mol_name = mol.GetProp("_Name")
            output_path = output_dir / f"{mol_name}.mol2"
            
            # Use OpenBabel for MOL2 conversion via temporary SDF
            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
                writer = Chem.SDWriter(tmp.name)
                writer.write(mol)
                writer.close()
                
                # Convert using OpenBabel
                obmol = next(pybel.readfile("sdf", tmp.name))
                
                # Ensure proper atom typing for MOL2
                obmol.OBMol.PerceiveBondOrders()
                obmol.OBMol.SetHybridizationPerceived()
                obmol.OBMol.SetAromaticPerceived()
                
                # Write MOL2
                obmol.write("mol2", str(output_path), overwrite=True)
                
                # Clean up temp file
                os.unlink(tmp.name)
            
            output_paths.append(output_path)
        
        logger.info(f"Saved {len(output_paths)} molecules to MOL2 format in {output_dir}")
        return output_paths