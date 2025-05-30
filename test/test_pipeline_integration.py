import pytest
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any

from HandsFreeDocking.Wrapper_Docking import PipelineDocking


class TestPipelineDockingIntegration:
    """Test the main PipelineDocking wrapper with multiple engines."""
    
    # @pytest.mark.slow
    # def test_single_engine_pipeline_docking(
    #     self,
    #     persistent_tmp_workdir,
    #     protein_pdb,
    #     crystal_sdf,
    #     ligands_sdf,
    #     single_docking_engine,
    #     toolkit,
    #     small_test_settings,
    #     output_validator
    # ):
    #     """Test PipelineDocking with a single docking engine."""
    #     n_confs, n_cpus = small_test_settings
        
    #     # Initialize PipelineDocking with single engine
    #     docking = PipelineDocking(
    #         workdir=persistent_tmp_workdir,
    #         docking_software=[single_docking_engine],
    #         settings=(n_confs, n_cpus),
    #         protein_pdb=protein_pdb,
    #         ligands_input=ligands_sdf,
    #         crystal_sdf=crystal_sdf,
    #         toolkit=toolkit
    #     )
        
    #     # Run the docking
    #     results = docking.run()
        
    #     # Validate results structure
    #     assert isinstance(results, dict), "Results should be a dictionary"
    #     assert single_docking_engine in results, f"Results should contain {single_docking_engine}"
        
    #     # Get concatenated DataFrame
    #     full_df = docking.concat_df()
    #     assert isinstance(full_df, pd.DataFrame), "concat_df should return a DataFrame"
    #     assert len(full_df) > 0, "DataFrame should not be empty"
        
    #     # Validate DataFrame columns
    #     expected_columns = ["ID", "Molecule", "Score", "Engine"]
    #     for col in expected_columns:
    #         assert col in full_df.columns, f"DataFrame should contain {col} column"
        
    #     # Validate engine-specific directory structure
    #     engine_dir = persistent_tmp_workdir / single_docking_engine.title()
    #     assert engine_dir.exists(), f"Engine directory {engine_dir} should exist"
        
    #     # Validate output files in engine directory
    #     output_dir = engine_dir / "output"
    #     assert output_dir.exists(), f"Output directory {output_dir} should exist"
        
    #     # Engine-specific file validation
    #     file_validation = output_validator["output_files"](engine_dir, single_docking_engine)
    #     engine_key = f"has_{single_docking_engine}_sdf_files" if single_docking_engine != "rxdock" else "has_sd_files"
    #     if engine_key in file_validation:
    #         assert file_validation[engine_key], f"Should have {single_docking_engine} output files"
        
    #     # Print info for manual inspection
    #     print(f"\n{'='*60}")
    #     print(f"🔧 Single Engine Test COMPLETED - {single_docking_engine.upper()}")
    #     print(f"{'='*60}")
    #     print(f"📁 Output Directory: {persistent_tmp_workdir}")
    #     print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
    #     print(f"📈 Results DataFrame shape: {full_df.shape}")
    #     print(f"📊 Number of poses: {len(full_df)}")
    #     print(f"🔍 Key directories to inspect:")
    #     print(f"   - Engine directory: {engine_dir}")
    #     print(f"   - Output files: {output_dir}")
    #     print(f"{'='*60}\n")
    
    @pytest.mark.slow
    def test_multi_engine_pipeline_docking(
        self,
        persistent_tmp_workdir,
        protein_pdb,
        crystal_sdf,
        ligands_sdf,
        multi_docking_engines,
        toolkit,
        small_test_settings,
        output_validator
    ):
        """Test PipelineDocking with multiple docking engines."""
        n_confs, n_cpus = small_test_settings
        
        # Initialize PipelineDocking with multiple engines
        docking = PipelineDocking(
            workdir=persistent_tmp_workdir,
            docking_software=multi_docking_engines,
            settings=(n_confs, n_cpus),
            protein_pdb=protein_pdb,
            ligands_input=ligands_sdf,
            crystal_sdf=crystal_sdf,
            toolkit=toolkit
        )
        
        # Run the docking
        results = docking.run()
        
        # Validate results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        
        # # Check that all engines are present in results
        # for engine in multi_docking_engines:
        #     assert engine in results, f"Results should contain {engine}"
        
        # Get concatenated DataFrame
        full_df = docking.concat_df()
        assert isinstance(full_df, pd.DataFrame), "concat_df should return a DataFrame"
        assert len(full_df) > 0, "DataFrame should not be empty"

        # TODO: Find a way to valide the story here.
        
        # # Validate that all engines contributed results
        # engines_in_df = full_df["Software"].unique()
        # for engine in multi_docking_engines:
        #     assert engine in engines_in_df, f"DataFrame should contain results from {engine}"
        
        # # Validate engine-specific directories exist
        # for engine in multi_docking_engines:
        #     engine_dir = persistent_tmp_workdir / engine.title()
        #     assert engine_dir.exists(), f"Engine directory {engine_dir} should exist"
            
        #     output_dir = engine_dir / "output"
        #     assert output_dir.exists(), f"Output directory {output_dir} should exist"
        
        # # Check score normalization (should be between 0 and 1)
        # scores = full_df["Score"]
        # assert scores.min() >= 0, "Normalized scores should be >= 0"
        # assert scores.max() <= 1, "Normalized scores should be <= 1"
        
        # Print info for manual inspection
        print(f"\n{'='*60}")
        print(f"🚀 Multi-Engine Integration Test COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {persistent_tmp_workdir}")
        print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
        print(f"⚙️ Toolkit used: {toolkit}")
        print(f"📈 Results DataFrame shape: {full_df.shape}")
        print(f"⚙️  Engines tested: {multi_docking_engines}")
        # print(f"✅ Engines in results: {list(engines_in_df)}")
        # print(f"📉 Score range: {scores.min():.3f} - {scores.max():.3f}")
        print(f"🔍 Key directories to inspect:")
        for engine in multi_docking_engines:
            engine_dir = persistent_tmp_workdir / engine.title()
            print(f"   - {engine.upper()}: {engine_dir}")
        print(f"{'='*60}\n")
    
    # @pytest.mark.slow
    # def test_pipeline_with_smiles_input(
    #     self,
    #     persistent_tmp_workdir,
    #     protein_pdb,
    #     crystal_sdf,
    #     toolkit,
    #     small_test_settings
    # ):
    #     """Test PipelineDocking with SMILES input instead of SDF."""
    #     n_confs, n_cpus = small_test_settings
        
    #     # Create a simple SMILES CSV file for testing
    #     smiles_file = persistent_tmp_workdir / "test_smiles.csv"
    #     smiles_data = pd.DataFrame({
    #         "SMILES": ["CCO", "CC(=O)O", "c1ccccc1"],
    #         "ID": ["ethanol", "acetic_acid", "benzene"]
    #     })
    #     smiles_data.to_csv(smiles_file, index=False)
        
    #     # Use only one engine for faster testing
    #     test_engine = ["rxdock"]  # RxDock is usually the most reliable
        
    #     # Initialize PipelineDocking with SMILES input
    #     docking = PipelineDocking(
    #         workdir=persistent_tmp_workdir,
    #         docking_software=test_engine,
    #         settings=(n_confs, n_cpus),
    #         protein_pdb=protein_pdb,
    #         ligands_input=smiles_file,
    #         crystal_sdf=crystal_sdf,
    #         toolkit=toolkit
    #     )
        
    #     # Run the docking
    #     results = docking.run()
        
    #     # Validate results
    #     assert isinstance(results, dict), "Results should be a dictionary"
    #     assert test_engine[0] in results, f"Results should contain {test_engine[0]}"
        
    #     # Get concatenated DataFrame
    #     full_df = docking.concat_df()
    #     assert isinstance(full_df, pd.DataFrame), "concat_df should return a DataFrame"
    #     assert len(full_df) > 0, "DataFrame should not be empty"
        
    #     # Check that molecules were properly processed from SMILES
    #     assert "Molecule" in full_df.columns, "DataFrame should contain Molecule column"
    #     assert full_df["Molecule"].notna().all(), "All molecules should be valid"
        
    #     # Print info for manual inspection
    #     print(f"\n{'='*60}")
    #     print(f"🧪 SMILES Input Test COMPLETED")
    #     print(f"{'='*60}")
    #     print(f"📁 Output Directory: {persistent_tmp_workdir}")
    #     print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
    #     print(f"📈 Results DataFrame shape: {full_df.shape}")
    #     print(f"📊 Processed {len(smiles_data)} SMILES inputs")
    #     print(f"🔍 Key files to inspect:")
    #     print(f"   - Input SMILES: {smiles_file}")
    #     print(f"   - Engine output: {persistent_tmp_workdir}/{test_engine[0].title()}")
    #     print(f"{'='*60}\n")
    
    # @pytest.mark.slow
    # def test_pipeline_result_persistence(
    #     self,
    #     persistent_tmp_workdir,
    #     protein_pdb,
    #     crystal_sdf,
    #     ligands_sdf,
    #     toolkit,
    #     small_test_settings
    # ):
    #     """Test that pipeline results can be saved and loaded."""
    #     n_confs, n_cpus = small_test_settings
        
    #     # Use single engine for speed
    #     test_engine = ["rxdock"]
        
    #     # Initialize and run PipelineDocking
    #     docking = PipelineDocking(
    #         workdir=persistent_tmp_workdir,
    #         docking_software=test_engine,
    #         settings=(n_confs, n_cpus),
    #         protein_pdb=protein_pdb,
    #         ligands_input=ligands_sdf,
    #         crystal_sdf=crystal_sdf,
    #         toolkit=toolkit
    #     )
        
    #     # Run the docking
    #     results = docking.run()
    #     full_df = docking.concat_df()
        
    #     # Save results to pickle file
    #     results_file = persistent_tmp_workdir / "docking_results.pkl"
    #     with open(results_file, "wb") as f:
    #         pickle.dump(full_df, f)
        
    #     # Validate pickle file was created and is not empty
    #     assert results_file.exists(), "Results pickle file should exist"
    #     assert results_file.stat().st_size > 0, "Results pickle file should not be empty"
        
    #     # Load and validate the pickled results
    #     with open(results_file, "rb") as f:
    #         loaded_df = pickle.load(f)
        
    #     assert isinstance(loaded_df, pd.DataFrame), "Loaded data should be a DataFrame"
    #     assert len(loaded_df) == len(full_df), "Loaded DataFrame should have same length"
    #     assert list(loaded_df.columns) == list(full_df.columns), "Loaded DataFrame should have same columns"
        
    #     # Print info for manual inspection
    #     print(f"\n{'='*60}")
    #     print(f"💾 Result Persistence Test COMPLETED")
    #     print(f"{'='*60}")
    #     print(f"📁 Output Directory: {persistent_tmp_workdir}")
    #     print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
    #     print(f"💾 Results saved to: {results_file}")
    #     print(f"📈 Original DataFrame shape: {full_df.shape}")
    #     print(f"📊 Loaded DataFrame shape: {loaded_df.shape}")
    #     print(f"🔍 Files to inspect:")
    #     print(f"   - Pickle file: {results_file}")
    #     print(f"   - Engine outputs: {persistent_tmp_workdir}/{test_engine[0].title()}")
    #     print(f"{'='*60}\n")


# class TestPipelineDockingConfiguration:
#     """Test different configuration options for PipelineDocking."""
    
#     def test_pipeline_initialization_parameters(
#         self,
#         tmp_workdir,
#         protein_pdb,
#         crystal_sdf,
#         ligands_sdf
#     ):
#         """Test PipelineDocking initialization with different parameters."""
        
#         # Test with minimal parameters
#         docking_minimal = PipelineDocking(
#             workdir=tmp_workdir,
#             docking_software=["rxdock"],
#             settings=(2, 1),
#             protein_pdb=protein_pdb,
#             ligands_input=ligands_sdf,
#             crystal_sdf=crystal_sdf,
#             toolkit="cdpkit"
#         )
        
#         assert docking_minimal.workdir == tmp_workdir
#         assert docking_minimal.docking_software == ["rxdock"]
#         assert docking_minimal.settings == (2, 1)
#         assert docking_minimal.toolkit == "cdpkit"
        
#         # Test with additional parameters
#         docking_full = PipelineDocking(
#             workdir=tmp_workdir,
#             docking_software=["rxdock", "plants"],
#             settings=(5, 2),
#             protein_pdb=protein_pdb,
#             ligands_input=ligands_sdf,
#             crystal_sdf=crystal_sdf,
#             toolkit="cdpkit",
#             protonation_method="protoss"
#         )
        
#         assert docking_full.docking_software == ["rxdock", "plants"]
#         assert docking_full.settings == (5, 2)
#         assert docking_full.protonation_method == "protoss"
        
#         print(f"\n{'='*60}")
#         print(f"⚙️  Configuration Test COMPLETED")
#         print(f"{'='*60}")
#         print(f"✅ Parameter validation successful")
#         print(f"{'='*60}\n")
    
#     def test_invalid_configuration_handling(
#         self,
#         tmp_workdir,
#         protein_pdb,
#         crystal_sdf,
#         ligands_sdf
#     ):
#         """Test how PipelineDocking handles invalid configurations."""
        
#         # Test with invalid docking software
#         with pytest.raises((ValueError, KeyError, AttributeError)):
#             PipelineDocking(
#                 workdir=tmp_workdir,
#                 docking_software=["invalid_engine"],
#                 settings=(2, 1),
#                 protein_pdb=protein_pdb,
#                 ligands_input=ligands_sdf,
#                 crystal_sdf=crystal_sdf,
#                 toolkit="cdpkit"
#             )
        
#         # Test with invalid toolkit
#         with pytest.raises((ValueError, ImportError, AttributeError)):
#             PipelineDocking(
#                 workdir=tmp_workdir,
#                 docking_software=["rxdock"],
#                 settings=(2, 1),
#                 protein_pdb=protein_pdb,
#                 ligands_input=ligands_sdf,
#                 crystal_sdf=crystal_sdf,
#                 toolkit="invalid_toolkit"
#             )
        
#         print(f"\n{'='*60}")
#         print(f"⚠️  Invalid Configuration Test COMPLETED")
#         print(f"{'='*60}")
#         print(f"✅ Error handling validation successful")
#         print(f"{'='*60}\n")


# class TestPipelineDockingOutputFormats:
#     """Test different output formats and data structures."""
    
#     @pytest.mark.slow
#     def test_dataframe_output_format(
#         self,
#         persistent_tmp_workdir,
#         protein_pdb,
#         crystal_sdf,
#         ligands_sdf,
#         toolkit,
#         small_test_settings
#     ):
#         """Test the format and content of output DataFrames."""
#         n_confs, n_cpus = small_test_settings
        
#         # Use single engine for focused testing
#         test_engine = ["rxdock"]
        
#         docking = PipelineDocking(
#             workdir=persistent_tmp_workdir,
#             docking_software=test_engine,
#             settings=(n_confs, n_cpus),
#             protein_pdb=protein_pdb,
#             ligands_input=ligands_sdf,
#             crystal_sdf=crystal_sdf,
#             toolkit=toolkit
#         )
        
#         results = docking.run()
#         full_df = docking.concat_df()
        
#         # Test DataFrame structure
#         assert isinstance(full_df, pd.DataFrame), "Output should be a DataFrame"
        
#         # Test required columns
#         required_cols = ["ID", "Molecule", "Score", "Engine"]
#         for col in required_cols:
#             assert col in full_df.columns, f"Required column {col} missing"
        
#         # Test data types
#         assert full_df["ID"].dtype == object, "ID column should be object/string type"
#         assert pd.api.types.is_numeric_dtype(full_df["Score"]), "Score column should be numeric"
#         assert full_df["Engine"].dtype == object, "Engine column should be object/string type"
        
#         # Test that molecules are RDKit molecule objects
#         from rdkit import Chem
#         assert all(isinstance(mol, Chem.rdchem.Mol) for mol in full_df["Molecule"]), "Molecule column should contain RDKit molecules"
        
#         # Test score normalization
#         scores = full_df["Score"]
#         assert scores.min() >= 0, "Normalized scores should be >= 0"
#         assert scores.max() <= 1, "Normalized scores should be <= 1"
        
#         print(f"\n{'='*60}")
#         print(f"📈 DataFrame Format Test COMPLETED")
#         print(f"{'='*60}")
#         print(f"📁 Output Directory: {persistent_tmp_workdir}")
#         print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
#         print(f"📈 DataFrame shape: {full_df.shape}")
#         print(f"📊 DataFrame columns: {list(full_df.columns)}")
#         print(f"📉 Score range: {scores.min():.3f} - {scores.max():.3f}")
#         print(f"{'='*60}\n")