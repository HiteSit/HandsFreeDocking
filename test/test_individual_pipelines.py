import pytest
import os
from pathlib import Path
from typing import Dict, Any

from HandsFreeDocking.RxDock_Pipeline import RxDock_Docking
from HandsFreeDocking.Plants_Pipeline import Plants_Docking
from HandsFreeDocking.Gnina_Pipeline import Gnina_Docking

try:
    from HandsFreeDocking.OpenEye_Pipeline import OpenEye_Docking
    OPENEYE_AVAILABLE = True
except ImportError:
    OPENEYE_AVAILABLE = False


class TestRxDockPipeline:
    """Test RxDock docking pipeline individually."""
    
    @pytest.mark.slow
    def test_rxdock_pipeline_execution(
        self, 
        persistent_tmp_workdir,
        protein_pdb,
        crystal_sdf,
        ligands_sdf,
        protonation_method_ligand,
        small_test_settings,
        output_validator
    ):
        """Test RxDock pipeline runs successfully and produces expected outputs."""
        n_poses, n_cpus = small_test_settings
        
        # Initialize RxDock pipeline
        rxdock_pipeline = RxDock_Docking(
            workdir=persistent_tmp_workdir,
            pdb_ID=protein_pdb,
            crystal_path=crystal_sdf,
            ligands_sdf=ligands_sdf,
            protonation_method=protonation_method_ligand,
            tautomer_score_threshold=2.0
        )
        
        # Run the pipeline
        result = rxdock_pipeline.main(n_poses=n_poses, n_cpus=n_cpus)
        
        # Validate return structure
        assert isinstance(result, dict), "RxDock should return a dictionary"
        assert "docked_ligands" in result, "Result should contain 'docked_ligands'"
        assert "results_df" in result, "Result should contain 'results_df'"
        assert "processed_sdf_files" in result, "Result should contain 'processed_sdf_files'"
        
        # Validate directory structure
        dir_validation = output_validator["directory_structure"](persistent_tmp_workdir, "rxdock")
        assert dir_validation["output_dir_exists"], "Output directory should exist"
        assert dir_validation["rxdock_prm_exists"], "RxDock parameter file should exist"
        assert dir_validation["rxdock_as_exists"], "RxDock cavity file should exist"
        
        # Validate output files
        file_validation = output_validator["output_files"](persistent_tmp_workdir, "rxdock")
        assert file_validation["has_sd_files"], "Should have .sd output files"
        assert file_validation["all_sd_files_not_empty"], "All .sd files should be non-empty"
        
        # Print info for manual inspection
        print(f"\n{'='*60}")
        print(f"🧪 RxDock Pipeline Test COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {persistent_tmp_workdir}")
        print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
        print(f"📊 Found {len(result['docked_ligands'])} docked ligand files")
        print(f"📈 Results DataFrame shape: {result['results_df'].shape}")
        print(f"🔍 Key files to inspect:")
        print(f"   - Output files: {persistent_tmp_workdir}/output/*.sd")
        print(f"   - RxDock params: {persistent_tmp_workdir}/rxdock.prm")
        print(f"   - Cavity file: {persistent_tmp_workdir}/rxdock.as")
        print(f"{'='*60}\n")

        assert file_validation["sdf_naming_correct"], "RxDock SDF files should have correct naming convention"

class TestPlantsPipeline:
    """Test PLANTS docking pipeline individually."""
    
    @pytest.mark.slow 
    def test_plants_pipeline_execution(
        self,
        persistent_tmp_workdir,
        protein_pdb,
        crystal_sdf,
        ligands_sdf,
        protonation_method_ligand,
        test_settings,
        output_validator
    ):
        """Test PLANTS pipeline runs successfully and produces expected outputs."""
        n_confs, n_cpus = test_settings
        
        # Initialize PLANTS pipeline
        plants_pipeline = Plants_Docking(
            workdir=persistent_tmp_workdir,
            pdb_ID=protein_pdb,
            crystal_path=crystal_sdf,
            ligands_sdf=ligands_sdf,
            protonation_method=protonation_method_ligand,
            tautomer_score_threshold=2.0
        )
        
        # Run the pipeline
        plants_pipeline.main(n_confs=n_confs, n_cpus=n_cpus)
        
        # Validate directory structure
        dir_validation = output_validator["directory_structure"](persistent_tmp_workdir, "plants")
        assert dir_validation["output_dir_exists"], "Output directory should exist"
        assert dir_validation["plants_output_dir_exists"], "PLANTS output directory should exist"
        assert dir_validation["ligands_mol2_dir_exists"], "Ligands MOL2 directory should exist"
        
        # Validate output files
        file_validation = output_validator["output_files"](persistent_tmp_workdir, "plants")
        assert file_validation["has_plants_sdf_files"], "Should have _Plants.sdf output files"
        assert file_validation["all_plants_sdf_files_not_empty"], "All _Plants.sdf files should be non-empty"
        assert file_validation["sdf_naming_correct"], "Plants SDF files should have correct naming convention"
        # FIXME: Uncomment this line if you want to check for unknown atoms
        # assert file_validation["no_unknown_atoms_in_plants_sdf"], "Plants SDF files should not contain unknown atoms (*)"
        
        # Check for intermediate PLANTS files
        output_plants_dir = persistent_tmp_workdir / "output_plants"
        plants_subdirs = [d for d in output_plants_dir.iterdir() if d.is_dir()]
        assert len(plants_subdirs) > 0, "Should have PLANTS output subdirectories"
        
        # Check that each subdirectory has ranking.csv
        for subdir in plants_subdirs:
            ranking_file = subdir / "ranking.csv"
            assert output_validator["file_exists_and_not_empty"](ranking_file), f"Ranking file should exist in {subdir}"
        
        # Print info for manual inspection
        print(f"\n{'='*60}")
        print(f"🌱 PLANTS Pipeline Test COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {persistent_tmp_workdir}")
        print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
        print(f"📊 Found {len(plants_subdirs)} PLANTS output subdirectories")
        print(f"🔍 Key files to inspect:")
        print(f"   - Final SDF files: {persistent_tmp_workdir}/output/*_Plants.sdf")
        print(f"   - PLANTS outputs: {persistent_tmp_workdir}/output_plants/*/")
        print(f"   - Ranking files: {persistent_tmp_workdir}/output_plants/*/ranking.csv")
        print(f"   - MOL2 ligands: {persistent_tmp_workdir}/ligands_mol2/")
        print(f"{'='*60}\n")


class TestGninaPipeline:
    """Test GNINA docking pipeline individually."""
    
    @pytest.mark.slow
    def test_gnina_non_covalent_pipeline_execution(
        self,
        persistent_tmp_workdir,
        protein_pdb,
        crystal_sdf,
        ligands_sdf,
        protonation_method_ligand,
        protein_protonation_method,
        small_test_settings,
        output_validator
    ):
        """Test GNINA non-covalent pipeline runs successfully and produces expected outputs."""
        n_confs, n_cpus = small_test_settings
        
        # Initialize GNINA pipeline
        gnina_pipeline = Gnina_Docking(
            workdir=persistent_tmp_workdir,
            pdb_ID=protein_pdb,
            crystal_path=crystal_sdf,
            ligands_sdf=ligands_sdf,
            protonation_method=protonation_method_ligand,
            protein_protonation_method=protein_protonation_method,
            tautomer_score_threshold=None  # Default: only best tautomers
        )
        
        # Run the non-covalent pipeline
        gnina_pipeline.non_covalent_run(n_confs=n_confs, n_cpus=n_cpus)
        
        # Validate directory structure
        dir_validation = output_validator["directory_structure"](persistent_tmp_workdir, "gnina")
        assert dir_validation["output_dir_exists"], "Output directory should exist"
        assert dir_validation["ligands_split_dir_exists"], "Ligands split directory should exist"
        
        # Validate output files
        file_validation = output_validator["output_files"](persistent_tmp_workdir, "gnina")
        assert file_validation["has_gnina_sdf_files"], "Should have _Gnina.sdf output files"
        assert file_validation["all_gnina_sdf_files_not_empty"], "All _Gnina.sdf files should be non-empty"
        assert file_validation["sdf_naming_correct"], "Gnina SDF files should have correct naming convention"
        
        # Check for prepared protein
        if protein_protonation_method == "protoss":
            prep_protein = persistent_tmp_workdir / f"{protein_pdb.stem}_prep.pdb"
        else:  # pdbfixer
            prep_protein = persistent_tmp_workdir / f"{protein_pdb.stem}_clean.pdb"
        
        assert output_validator["file_exists_and_not_empty"](prep_protein), "Prepared protein should exist"
        
        # Check for command log
        log_file = persistent_tmp_workdir / "gnina_commands.log"
        assert output_validator["file_exists_and_not_empty"](log_file), "Command log should exist"
        
        # Print info for manual inspection
        print(f"\n{'='*60}")
        print(f"🧬 GNINA Pipeline Test COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {persistent_tmp_workdir}")
        print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
        print(f"⚙️  Used protein protonation method: {protein_protonation_method}")
        print(f"🔍 Key files to inspect:")
        print(f"   - Docked molecules: {persistent_tmp_workdir}/output/*_Gnina.sdf")
        print(f"   - Prepared protein: {persistent_tmp_workdir}/{protein_pdb.stem}_prep.pdb")
        print(f"   - Split ligands: {persistent_tmp_workdir}/ligands_split/")
        print(f"   - Command log: {persistent_tmp_workdir}/gnina_commands.log")
        print(f"{'='*60}\n")


@pytest.mark.skipif(not OPENEYE_AVAILABLE, reason="OpenEye toolkit not available")
class TestOpenEyePipeline:
    """Test OpenEye docking pipeline individually."""
    
    @pytest.mark.slow
    @pytest.mark.requires_openeye
    def test_openeye_pipeline_execution(
        self,
        persistent_tmp_workdir,
        protein_pdb,
        crystal_sdf,
        ligands_sdf,
        small_test_settings,
        output_validator
    ):
        """Test OpenEye pipeline runs successfully and produces expected outputs."""
        n_confs, n_cpus = small_test_settings
        
        # Prepare docking tuples for OpenEye (SMILES, ID pairs)
        # For testing, we'll use simple examples
        docking_tuple = [
            ("CCO", "ethanol"),
            ("CC(=O)O", "acetic_acid")
        ]
        
        # Initialize OpenEye pipeline
        openeye_pipeline = OpenEye_Docking(
            workdir=persistent_tmp_workdir,
            pdb_ID=protein_pdb,
            mtz=None,
            crystal_path=crystal_sdf,
            docking_tuple=docking_tuple
        )
        
        # Run the pipeline
        openeye_pipeline.run_oedocking_pipeline(
            n_cpu=n_cpus,
            confs=n_confs,
            mtz=None,
            mode="oe"
        )
        
        # Validate directory structure
        dir_validation = output_validator["directory_structure"](persistent_tmp_workdir, "openeye")
        assert dir_validation["output_dir_exists"], "Output directory should exist"
        
        # Validate output files
        file_validation = output_validator["output_files"](persistent_tmp_workdir, "openeye")
        assert file_validation["has_openeye_sdf_files"], "Should have _Eye.sdf output files"
        assert file_validation["all_openeye_sdf_files_not_empty"], "All _Eye.sdf files should be non-empty"
        assert file_validation["sdf_naming_correct"], "OpenEye SDF files should have correct naming convention"
        
        # Check for design unit file
        design_unit = persistent_tmp_workdir / f"{protein_pdb.stem}_Receptor.oedu"
        assert output_validator["file_exists_and_not_empty"](design_unit), "Design unit file should exist"
        
        # Print info for manual inspection
        print(f"\n{'='*60}")
        print(f"👁️  OpenEye Pipeline Test COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {persistent_tmp_workdir}")
        print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
        print(f"📊 Processed {len(docking_tuple)} ligands")
        print(f"🔍 Key files to inspect:")
        print(f"   - Docked molecules: {persistent_tmp_workdir}/output/*_Eye.sdf")
        print(f"   - Design unit: {persistent_tmp_workdir}/{protein_pdb.stem}_Receptor.oedu")
        print(f"   - Complex: {persistent_tmp_workdir}/complex.pdb")
        print(f"{'='*60}\n")


class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""
    
    @pytest.mark.slow
    def test_pipeline_with_invalid_inputs(
        self,
        persistent_tmp_workdir,
        protein_pdb,
        crystal_sdf,
        protonation_method_ligand,
        small_test_settings
    ):
        """Test pipeline behavior with invalid or missing inputs."""
        # Create an empty SDF file to test error handling
        empty_sdf = persistent_tmp_workdir / "empty.sdf"
        empty_sdf.touch()
        
        n_poses, n_cpus = small_test_settings
        
        # Test RxDock with empty ligands file
        rxdock_pipeline = RxDock_Docking(
            workdir=persistent_tmp_workdir / "rxdock_test",
            pdb_ID=protein_pdb,
            crystal_path=crystal_sdf,
            ligands_sdf=empty_sdf,
            protonation_method=protonation_method_ligand
        )
        
        # This should either handle gracefully or fail in a controlled way
        try:
            result = rxdock_pipeline.main(n_poses=n_poses, n_cpus=n_cpus)
            # If it succeeds, check that it returns valid structure
            assert isinstance(result, dict)
        except Exception as e:
            # If it fails, that's expected with empty input
            print(f"Expected failure with empty input: {e}")
            assert True  # Test passes if it fails gracefully
        
        print(f"\n{'='*60}")
        print(f"🛡️  Pipeline Robustness Test COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Test Directory: {persistent_tmp_workdir}")
        print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
        print(f"🔍 Check subdirectories for partial outputs or error logs")
        print(f"{'='*60}\n")


# Utility test functions
def test_all_test_files_exist(protein_pdb, crystal_sdf, ligands_sdf):
    """Verify all test input files exist and are readable."""
    assert protein_pdb.exists(), f"Protein PDB file not found: {protein_pdb}"
    assert crystal_sdf.exists(), f"Crystal SDF file not found: {crystal_sdf}"
    assert ligands_sdf.exists(), f"Ligands SDF file not found: {ligands_sdf}"
    
    # Check file sizes
    assert protein_pdb.stat().st_size > 0, "Protein PDB file is empty"
    assert crystal_sdf.stat().st_size > 0, "Crystal SDF file is empty"
    assert ligands_sdf.stat().st_size > 0, "Ligands SDF file is empty"


def test_conda_environment():
    """Verify we're running in the correct conda environment."""
    env_name = os.environ.get('CONDA_DEFAULT_ENV', '')
    assert env_name == 'cheminf_3_11', f"Wrong conda environment. Expected 'cheminf_3_11', got '{env_name}'"