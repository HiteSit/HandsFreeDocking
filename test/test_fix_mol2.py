import pytest
from pathlib import Path

import datamol as dm
from rdkit import Chem

from HandsFreeDocking.tools.Fix_Mol2 import hard_fix_mol2


class TestFixMol2:
    """Test the hard_fix_mol2 function for fixing broken mol2 files."""

    def test_hard_fix_mol2_function(
        self,
        mol2_broken_files,
        persistent_tmp_workdir
    ):
        """Test hard_fix_mol2 function fixes all poses from broken mol2 files."""
        
        # Test each mol2 file in the fixture
        for mol2_path, template_smiles in mol2_broken_files.items():
            mol2_file = Path(mol2_path).name
            
            assert Path(mol2_path).exists(), f"Mol2 file does not exist: {mol2_path}"
            
            # Read the broken molecules first to count expected poses
            broken_mols = dm.read_mol2file(mol2_path, sanitize=False)
            n_broken_poses = len(broken_mols)
            
            print(f"\n=== Testing {mol2_file} ===")
            print(f"Template SMILES: {template_smiles}")
            print(f"Found {n_broken_poses} poses in broken mol2 file")
            
            # Test 1: Get fixed molecules without writing to file
            fixed_mols = hard_fix_mol2(
                mol2_filename=mol2_path,
                template_smiles=template_smiles,
                output_sdf=None  # Return molecules instead of writing
            )
            
            # Validate return type and completeness
            assert isinstance(fixed_mols, list), "hard_fix_mol2 should return a list when output_sdf=None"
            assert len(fixed_mols) == n_broken_poses, f"Expected {n_broken_poses} fixed molecules, got {len(fixed_mols)}"
            
            # Validate ALL molecules are fixed (no None values)
            none_count = sum(1 for mol in fixed_mols if mol is None)
            assert none_count == 0, f"ALL poses must be fixed! Found {none_count} unfixed poses out of {n_broken_poses}"
            
            # Validate all molecules can be sanitized
            for i, mol in enumerate(fixed_mols):
                assert mol is not None, f"Pose {i+1} was not fixed"
                try:
                    Chem.SanitizeMol(mol)
                    print(f"  ✓ Pose {i+1}: Successfully sanitized")
                except Exception as e:
                    pytest.fail(f"Pose {i+1} failed sanitization: {str(e)}")
            
            # Test 2: Write to SDF file for manual inspection
            output_sdf = persistent_tmp_workdir / f"{Path(mol2_file).stem}_fixed.sdf"
            
            # Call with output file (returns None)
            result = hard_fix_mol2(
                mol2_filename=mol2_path,
                template_smiles=template_smiles,
                output_sdf=str(output_sdf)
            )
            
            # Should return None when output_sdf is provided
            assert result is None, "hard_fix_mol2 should return None when output_sdf is provided"
            
            # Validate SDF file was created and is not empty
            assert output_sdf.exists(), f"Output SDF file was not created: {output_sdf}"
            assert output_sdf.stat().st_size > 0, f"Output SDF file is empty: {output_sdf}"
            
            # Read back the SDF to verify content
            sdf_mols = [mol for mol in Chem.SDMolSupplier(str(output_sdf)) if mol is not None]
            assert len(sdf_mols) == n_broken_poses, f"SDF should contain {n_broken_poses} molecules, found {len(sdf_mols)}"
            
            print(f"  ✓ All {n_broken_poses} poses successfully fixed and written to SDF")
            print(f"  📁 Output written to: {output_sdf}")
            print(f"  📁 Temp directory: {persistent_tmp_workdir}")
            
            print(f"=== {mol2_file} test completed successfully ===\n")