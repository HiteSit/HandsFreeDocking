import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import os

# Fixtures for test inputs
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent.parent / "examples"

@pytest.fixture(scope="session") 
def protein_pdb(test_data_dir) -> Path:
    """Test protein PDB file."""
    return test_data_dir / "LAG3_Moloc_2.pdb"

@pytest.fixture(scope="session")
def ligands_sdf(test_data_dir) -> Path:
    """Test ligands SDF file.""" 
    return test_data_dir / "Ligands_To_Dock.sdf"

@pytest.fixture(scope="session")
def crystal_sdf(test_data_dir) -> Path:
    """Test crystal ligand SDF file."""
    return test_data_dir / "Fake_Crystal.sdf"

# Fixtures for test configuration
@pytest.fixture(scope="session")
def test_settings() -> Tuple[int, int]:
    """Test settings for docking (n_conformers, n_cpus)."""
    return (5, 2)  # Reduced for faster testing

@pytest.fixture(scope="session")
def small_test_settings() -> Tuple[int, int]:
    """Minimal test settings for very fast testing."""
    return (2, 1)

# Fixtures for temporary directories
@pytest.fixture
def tmp_workdir():
    """Create a temporary working directory for each test."""
    with tempfile.TemporaryDirectory(prefix="test_docking_") as tmp_dir:
        workdir = Path(tmp_dir)
        yield workdir
        # Cleanup is automatic with TemporaryDirectory

@pytest.fixture
def persistent_tmp_workdir(tmp_path):
    """Create a temporary directory that persists for manual inspection."""
    # Use pytest's tmp_path which creates a unique directory per test
    # This will be in /tmp/pytest-of-<user>/pytest-current/<test-name>
    workdir = tmp_path / "docking_output"
    workdir.mkdir(exist_ok=True)
    return workdir

# Fixtures for docking software configuration
@pytest.fixture(params=["cdp", "oe"])
def protonation_method_ligand(request) -> str:
    """Parameterized fixture for different ligand protonation methods."""
    method_name = request.param
    
    # Skip OpenEye tests if not available
    if method_name == "oe":
        try:
            from openeye import oechem
        except ImportError:
            pytest.skip("OpenEye toolkit not available")
    
    return method_name

@pytest.fixture(params=["protoss", "pdbfixer"])
def protein_protonation_method(request) -> str:
    """Parameterized fixture for different protein protonation methods."""
    return request.param

# Fixtures for individual docking engines
@pytest.fixture(params=["rxdock", "plants", "gnina"])
def single_docking_engine(request) -> str:
    """Parameterized fixture for testing individual docking engines."""
    return request.param

@pytest.fixture
def multi_docking_engines() -> list:
    """Fixture for testing multiple docking engines together."""
    return ["rxdock", "plants"]

@pytest.fixture
def all_docking_engines() -> list:
    """Fixture for testing all available docking engines."""
    engines = ["rxdock", "plants", "gnina"]
    
    # Add OpenEye if available
    try:
        from openeye import oechem
        engines.append("openeye")
    except ImportError:
        pass
    
    return engines

# Fixtures for validation functions
@pytest.fixture
def output_validator():
    """Fixture providing file validation functions."""
    
    def validate_file_exists_and_not_empty(file_path: Path) -> bool:
        """Check if file exists and is not empty."""
        return file_path.exists() and file_path.stat().st_size > 0
    
    def validate_directory_structure(workdir: Path, engine: str) -> Dict[str, bool]:
        """Validate expected directory structure for a docking engine."""
        results = {}
        
        # Common directories
        output_dir = workdir / "output"
        results["output_dir_exists"] = output_dir.exists()
        
        # Engine-specific checks
        if engine == "rxdock":
            results["rxdock_prm_exists"] = validate_file_exists_and_not_empty(workdir / "rxdock.prm")
            results["rxdock_as_exists"] = validate_file_exists_and_not_empty(workdir / "rxdock.as")
        elif engine == "plants":
            plants_output_dir = workdir / "output_plants"
            results["plants_output_dir_exists"] = plants_output_dir.exists()
            results["ligands_mol2_dir_exists"] = (workdir / "ligands_mol2").exists()
        elif engine == "gnina":
            ligands_split_dir = workdir / "ligands_split"
            results["ligands_split_dir_exists"] = ligands_split_dir.exists()
        
        return results
    
    def validate_output_files(workdir: Path, engine: str) -> Dict[str, bool]:
        """Validate expected output files for a docking engine."""
        results = {}
        output_dir = workdir / "output"
        
        if not output_dir.exists():
            return {"output_dir_missing": True}
        
        # Look for output files with engine-specific extensions
        if engine == "rxdock":
            # RxDock creates .sd files
            sd_files = list(output_dir.glob("*.sd"))
            results["has_sd_files"] = len(sd_files) > 0
            results["all_sd_files_not_empty"] = all(f.stat().st_size > 0 for f in sd_files)
        elif engine == "plants":
            # Plants creates _Plants.sdf files
            sdf_files = list(output_dir.glob("*_Plants.sdf"))
            results["has_plants_sdf_files"] = len(sdf_files) > 0
            results["all_plants_sdf_files_not_empty"] = all(f.stat().st_size > 0 for f in sdf_files)
        elif engine == "gnina":
            # Gnina creates _Gnina.sdf files
            sdf_files = list(output_dir.glob("*_Gnina.sdf"))
            results["has_gnina_sdf_files"] = len(sdf_files) > 0
            results["all_gnina_sdf_files_not_empty"] = all(f.stat().st_size > 0 for f in sdf_files)
        elif engine == "openeye":
            # OpenEye creates _Eye.sdf files
            sdf_files = list(output_dir.glob("*_Eye.sdf"))
            results["has_openeye_sdf_files"] = len(sdf_files) > 0
            results["all_openeye_sdf_files_not_empty"] = all(f.stat().st_size > 0 for f in sdf_files)
        
        return results
    
    return {
        "file_exists_and_not_empty": validate_file_exists_and_not_empty,
        "directory_structure": validate_directory_structure,
        "output_files": validate_output_files
    }

# Environment setup fixture
@pytest.fixture(scope="session", autouse=True)
def setup_conda_env():
    """Ensure the correct conda environment is activated."""
    # Check if we're in the right environment
    env_name = os.environ.get('CONDA_DEFAULT_ENV', '')
    if env_name != 'cheminf_3_11':
        pytest.skip("Tests require 'cheminf_3_11' conda environment to be activated")

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires external docking software)"
    )
    config.addinivalue_line(
        "markers", "requires_openeye: mark test as requiring OpenEye toolkit"
    )
    config.addinivalue_line(
        "markers", "requires_external_software: mark test as requiring external docking software"
    )

def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their requirements."""
    for item in items:
        # Mark tests that use external docking software as slow
        if any(engine in item.name.lower() for engine in ["rxdock", "plants", "gnina", "openeye"]):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_external_software)
        
        # Mark OpenEye-specific tests
        if "openeye" in item.name.lower():
            item.add_marker(pytest.mark.requires_openeye)