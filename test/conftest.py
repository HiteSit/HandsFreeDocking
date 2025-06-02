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
    return test_data_dir / "Ligands_Complex.sdf"

@pytest.fixture(scope="session")
def crystal_sdf(test_data_dir) -> Path:
    """Test crystal ligand SDF file."""
    return test_data_dir / "Fake_Crystal.sdf"

@pytest.fixture(scope="session")
def mol2_broken_files(test_data_dir) -> Dict[str, str]:
    """Mapping of broken mol2 files to their template SMILES strings."""
    mol2_dir = test_data_dir / "Mol2_Broken"
    
    # Map mol2 files to their known SMILES strings
    # These are well-known flavonoid natural products
    return {
        str(mol2_dir / "6-methylflavone_Broken.mol2"): "CC1=CC2=C(C=C1)OC(=CC2=O)C3=CC=CC=C3",
        str(mol2_dir / "Apigenin_Broken.mol2"): "C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O",
        str(mol2_dir / "Chrysin_Broken.mol2"): "C1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O",
        str(mol2_dir / "Eriodictyol_Broken.mol2"): "C1C(OC2=CC(=CC(=C2C1=O)O)O)C3=CC(=C(C=C3)O)O",
        str(mol2_dir / "Fisetin_Broken.mol2"): "C1=CC(=C(C=C1C2=C(C(=O)C3=C(O2)C=C(C=C3)O)O)O)O",
    }

# Fixtures for test configuration
@pytest.fixture(scope="session")
def test_settings() -> Tuple[int, int]:
    """Test settings for docking (n_conformers, n_cpus)."""
    return (5, 8)  # Default: 5 conformers, 8 CPUs

@pytest.fixture(scope="session")
def small_test_settings() -> Tuple[int, int]:
    """Minimal test settings for very fast testing."""
    return (2, 1)  # Default: 2 conformers, 1 CPU

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
@pytest.fixture(params=["oe"])
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

@pytest.fixture(params=[None, 2])
def tautomer_score_threshold(request) -> Any:
    """Parameterized fixture for tautomer score threshold."""
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
    engines = ["rxdock", "plants", "gnina", "openeye"]
    
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
    
    def has_unknown_atoms_in_sdf(sdf_file_path: Path) -> bool:
        """
        Check if any molecule in an SDF file contains unknown atoms (*).
        
        Parameters:
        -----------
        sdf_file_path : Path
            Path to the SDF file
            
        Returns:
        --------
        bool
            True if any molecule contains unknown atoms, False otherwise
        """
        from rdkit import Chem
        
        # Create supplier without sanitization
        supplier = Chem.SDMolSupplier(str(sdf_file_path), sanitize=False)
        
        for mol_idx, mol in enumerate(supplier):
            # Skip if molecule couldn't be parsed
            if mol is None:
                continue
                
            # Check each atom in the molecule
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:  # Atomic number 0 indicates a dummy/unknown atom
                    return True
        
        return False

    def validate_sdf_naming(sdf_file_path: Path, software: str) -> bool:
        """
        Validate SDF file naming convention for both filename and internal molecule names.
        
        Expected patterns:
        - Filename: {LIG_NAME}_Iso{NUM}_Taut{NUM}_{SOFTWARE}.sdf (or .sd for rxdock)
        - Internal names: {LIG_NAME}_Iso{NUM}_Taut{NUM}_{SOFTWARE}-P{POSE_NUM}
        
        Parameters:
        -----------
        sdf_file_path : Path
            Path to the SDF file
        software : str
            Software name ("rxdock", "plants", "gnina", "openeye")
            
        Returns:
        --------
        bool
            True if naming conventions are correct, False otherwise
        """
        import re
        from rdkit import Chem
        
        # Define filename patterns per software
        # Support both tautomer enumeration cases:
        # - With tautomers: {BASE}_Iso{N}_Taut{M}_{SOFTWARE}
        # - Without tautomers: {BASE}_Iso{N}_{SOFTWARE}
        filename_patterns = {
            "rxdock": r".*_Iso\d+(_Taut\d+)?_Rxdock\.sd$",
            "plants": r".*_Iso\d+(_Taut\d+)?_Plants\.sdf$", 
            "gnina": r".*_Iso\d+(_Taut\d+)?_Gnina\.sdf$",
            "openeye": r".*_Iso\d+(_Taut\d+)?_Eye\.sdf$"
        }
        
        # Define internal name patterns per software
        # Support both tautomer enumeration cases:
        # - With tautomers: {BASE}_Iso{N}_Taut{M}_{SOFTWARE}-P{POSE}
        # - Without tautomers: {BASE}_Iso{N}_{SOFTWARE}-P{POSE}
        internal_patterns = {
            "rxdock": r".*_Iso\d+(_Taut\d+)?_Rxdock-P\d+$",
            "plants": r".*_Iso\d+(_Taut\d+)?_Plants-P\d+$",
            "gnina": r".*_Iso\d+(_Taut\d+)?_Gnina-P\d+$", 
            "openeye": r".*_Iso\d+(_Taut\d+)?_Eye-P\d+$"
        }
        
        if software not in filename_patterns:
            return False
            
        # Validate filename
        filename = sdf_file_path.name
        if not re.match(filename_patterns[software], filename):
            return False
            
        # Validate internal molecule names
        try:
            supplier = Chem.SDMolSupplier(str(sdf_file_path), sanitize=False)
            
            for mol in supplier:
                if mol is None:
                    continue
                    
                # Get molecule name
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                
                # Check if name matches pattern
                if not re.match(internal_patterns[software], mol_name):
                    return False
                    
        except Exception:
            # If we can't read the file, consider it invalid naming
            return False
            
        return True

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
            results["sdf_naming_correct"] = all(validate_sdf_naming(f, "rxdock") for f in sd_files)
        elif engine == "plants":
            # Plants creates _Plants.sdf files
            sdf_files = list(output_dir.glob("*_Plants.sdf"))
            results["has_plants_sdf_files"] = len(sdf_files) > 0
            results["all_plants_sdf_files_not_empty"] = all(f.stat().st_size > 0 for f in sdf_files)
            results["sdf_naming_correct"] = all(validate_sdf_naming(f, "plants") for f in sdf_files)
            # Check for unknown atoms in ALL Plants SDF files
            # results["no_unknown_atoms_in_plants_sdf"] = all(
            #     not has_unknown_atoms_in_sdf(sdf_file) for sdf_file in sdf_files
            # )
        elif engine == "gnina":
            # Gnina creates _Gnina.sdf files
            sdf_files = list(output_dir.glob("*_Gnina.sdf"))
            results["has_gnina_sdf_files"] = len(sdf_files) > 0
            results["all_gnina_sdf_files_not_empty"] = all(f.stat().st_size > 0 for f in sdf_files)
            results["sdf_naming_correct"] = all(validate_sdf_naming(f, "gnina") for f in sdf_files)
        elif engine == "openeye":
            # OpenEye creates _Eye.sdf files
            sdf_files = list(output_dir.glob("*_Eye.sdf"))
            results["has_openeye_sdf_files"] = len(sdf_files) > 0
            results["all_openeye_sdf_files_not_empty"] = all(f.stat().st_size > 0 for f in sdf_files)
            results["sdf_naming_correct"] = all(validate_sdf_naming(f, "openeye") for f in sdf_files)
        
        return results
    
    return {
        "file_exists_and_not_empty": validate_file_exists_and_not_empty,
        "directory_structure": validate_directory_structure,
        "output_files": validate_output_files,
        "sdf_naming": validate_sdf_naming
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