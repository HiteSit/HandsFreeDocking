import pytest
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any

from HandsFreeDocking.Wrapper_Docking import PipelineDocking


class TestPipelineDockingIntegration:
    """Test the main PipelineDocking wrapper with multiple engines."""
    
    @pytest.mark.slow
    def test_multi_engine_pipeline_docking(
        self,
        persistent_tmp_workdir,
        protein_pdb,
        crystal_sdf,
        ligands_sdf,
        multi_docking_engines,
        protonation_method_ligand_oe_only,
        tautomer_score_threshold,
        test_settings,
        output_validator
    ):
        """Test PipelineDocking with multiple docking engines."""
        n_confs, n_cpus = test_settings
        
        # Initialize PipelineDocking with multiple engines
        docking = PipelineDocking(
            workdir=persistent_tmp_workdir,
            docking_software=multi_docking_engines,
            settings=(n_confs, n_cpus),
            protein_pdb=protein_pdb,
            ligands_input=ligands_sdf,
            crystal_sdf=crystal_sdf,
            protonation_method=protonation_method_ligand_oe_only,
            tautomer_score_threshold=tautomer_score_threshold
        )
        
        # Run the docking
        results = docking.run()
        
        # Validate results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        
        # Check that all engines are present in results
        for engine in multi_docking_engines:
            assert engine in results, f"Results should contain {engine}"
        
        # Get concatenated DataFrame
        full_df = docking.concat_df()
        assert isinstance(full_df, pd.DataFrame), "concat_df should return a DataFrame"
        assert len(full_df) > 0, "DataFrame should not be empty"
        
        # Validate that all engines contributed results
        engines_in_df = full_df["Software"].unique()
        for engine in multi_docking_engines:
            assert engine in engines_in_df, f"DataFrame should contain results from {engine}"

        # Validate the software into the DataFrame
        software_from_lst = sorted(list(full_df["Software"].unique()))
        assert software_from_lst == sorted(multi_docking_engines), "Software in DataFrame should match input engines"
        
        # Validate engine-specific directories exist
        for engine in multi_docking_engines:
            
            engine_snake = engine.title()
            engine_dir = persistent_tmp_workdir / engine_snake
            assert engine_dir.exists(), f"Engine directory {engine_dir} should exist"

            dir_validation = output_validator["directory_structure"](engine_dir, engine)
            # Check if all the items of the dictionary are True
            for key, value in dir_validation.items():
                assert value, f"Should have {key} directory in {engine_snake} directory"
            
            file_validation = output_validator["output_files"](engine_dir, engine)
            # Check if all the items of the dictionary are True
            for key, value in file_validation.items():
                assert value, f"Should have {key} files in {engine_snake} directory"

        # Save the FULL_DF in CSV
        full_df_path = persistent_tmp_workdir / "full_docking_results.csv"
        full_df.to_csv(full_df_path, index=False)
        
        # Print info for manual inspection
        print(f"\n{'='*60}")
        print(f"🚀 Multi-Engine Integration Test COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {persistent_tmp_workdir}")
        print(f"📋 Copy-paste path: {persistent_tmp_workdir}")
        print(f"⚙️ Ligand protonation method used: {protonation_method_ligand_oe_only}")
        print(f"📈 Results DataFrame shape: {full_df.shape}")
        print(f"⚙️ Engines tested: {multi_docking_engines}")
        print(f"✅ Engines in results: {list(engines_in_df)}")
        print(f"🔍 Key directories to inspect:")
        for engine in multi_docking_engines:
            engine_dir = persistent_tmp_workdir / engine.title()
            print(f"   - {engine.upper()}: {engine_dir}")
        print(f"📂 Full results saved to: {full_df_path}")
        print(f"{'='*60}\n")