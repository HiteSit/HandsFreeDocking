# HandsFreeDocking Test Suite

This directory contains pytest tests for the HandsFreeDocking molecular docking pipeline. The tests are designed to validate that the docking pipelines run successfully and produce expected outputs without validating the scientific correctness of the results.

## 🚀 **Quick Start - Most Important Commands**

```bash
# 1. Activate environment
conda activate cheminf_3_11

# 2. Run tests with output paths (MOST IMPORTANT!)
pytest test/ -s -v

# 3. Run just one pipeline test
pytest test/test_individual_pipelines.py::TestRxDockPipeline -s -v

# 4. Skip slow tests (for quick validation)
pytest test/ -m "not slow" -s -v
```

**The `-s` flag is KEY - it shows you exactly where the output files are saved!**

## Test Philosophy

These tests follow a **flexible validation approach**:
- ✅ **Check that pipelines run without crashing**
- ✅ **Verify expected files are created and non-empty**
- ✅ **Validate directory structure and basic data formats**
- ❌ **Do NOT validate scientific accuracy of docking results**
- 📁 **Generate outputs in temporary directories for manual inspection**

## Prerequisites

1. **Conda Environment**: Ensure the `cheminf_3_11` environment is activated
   ```bash
   conda activate cheminf_3_11
   ```

2. **External Software**: Install required docking software:
   - **RxDock**: Set `RBT_ROOT` environment variable
   - **PLANTS**: Set `PLANTS_HOME` environment variable  
   - **GNINA**: Available in PATH
   - **OpenEye** (optional): Valid license required
   - **Protoss**: For protein preparation

3. **Test Data**: The tests use example files in `../examples/`:
   - `LAG3_Moloc_2.pdb` - Test protein structure
   - `Ligands_To_Dock.sdf` - Test ligands
   - `Fake_Crystal.sdf` - Test crystal ligand

## Running Tests

### Quick Setup
```bash
# 1. Activate the conda environment
conda activate cheminf_3_11

# 2. Navigate to the project directory
cd /path/to/HandsFreeDocking
```

### 🚀 **RECOMMENDED: Run with Output Display**
```bash
# Run tests with detailed output paths (RECOMMENDED)
pytest test/ -s -v

# Run specific pipeline with detailed output
pytest test/test_individual_pipelines.py::TestRxDockPipeline -s -v

# Run only fast tests with output paths
pytest test/ -m "not slow" -s -v
```

### Basic Test Commands
```bash
# Run all tests (quiet mode)
pytest test/

# Run only fast tests (skip slow integration tests)
pytest test/ -m "not slow"

# Run tests for specific pipeline
pytest test/test_individual_pipelines.py::TestRxDockPipeline -v

# Run with specific toolkit
pytest test/ -k "cdpkit" -v
```

### Full Integration Tests
```bash
# Run all tests including slow integration tests WITH OUTPUT PATHS
pytest test/ -s -v --tb=short

# Run only integration tests with output display
pytest test/test_pipeline_integration.py -s -v

# Run with parallel execution (if pytest-xdist installed)
pytest test/ -n auto
```

### Test with Different Configurations
```bash
# Test with OpenEye toolkit (if available) - with output paths
pytest test/ -k "openeye" -s -v

# Test specific docking engines - with output paths
pytest test/ -k "rxdock" -s -v
pytest test/ -k "plants" -s -v
pytest test/ -k "gnina" -s -v
```

### 📁 **Understanding Test Output with `-s` Flag**

When you run tests with the `-s` flag, you'll see colorful output like this:

```
============================================================
🧪 RxDock Pipeline Test COMPLETED
============================================================
📁 Output Directory: /tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/
📋 Copy-paste path: /tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/
📊 Found 8 docked ligand files
📈 Results DataFrame shape: (40, 6)
🔍 Key files to inspect:
   - Output files: /tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/output/*.sd
   - RxDock params: /tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/rxdock.prm
   - Cavity file: /tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/rxdock.as
============================================================
```

**Simply copy-paste the path shown after "📋 Copy-paste path:" to explore the results!**

## Test Structure

### `conftest.py`
Central configuration file containing:
- **Shared fixtures** for test data, temporary directories, and validation functions
- **Parameterized fixtures** for different toolkits and engines
- **Environment validation** to ensure correct conda environment
- **Custom pytest markers** for test categorization

### `test_individual_pipelines.py`
Tests for individual docking engine pipelines:
- `TestRxDockPipeline` - RxDock docking tests
- `TestPlantsPipeline` - PLANTS docking tests  
- `TestGninaPipeline` - GNINA docking tests
- `TestOpenEyePipeline` - OpenEye docking tests (if available)
- `TestPipelineRobustness` - Error handling and edge cases

### `test_pipeline_integration.py`
Tests for the main `PipelineDocking` wrapper:
- `TestPipelineDockingIntegration` - Multi-engine workflow tests
- `TestPipelineDockingConfiguration` - Parameter validation tests
- `TestPipelineDockingOutputFormats` - Output format validation tests

## Test Output

### Temporary Directories
Tests create temporary directories for outputs:
- **Location**: `/tmp/pytest-of-<user>/pytest-current/<test-name>/`
- **Content**: Complete docking pipeline outputs
- **Persistence**: Directories remain after tests for manual inspection

### Validation Results
Tests validate:
- ✅ Output directories exist (`workdir/output/`)
- ✅ Engine-specific files are created and non-empty
- ✅ Expected file formats (`.sdf`, `.sd`, `.mol2`, etc.)
- ✅ DataFrame structure and data types
- ✅ Score normalization (0-1 range)

### Example Output Structure
```
/tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/
├── output/
│   ├── ligand1.sd
│   ├── ligand2.sd
│   └── ...
├── rxdock.prm
├── rxdock.as
├── ligands_split/
└── ...
```

## Test Markers

Tests use custom markers for organization:
- `@pytest.mark.slow` - Integration tests requiring external software
- `@pytest.mark.requires_openeye` - Tests requiring OpenEye toolkit
- `@pytest.mark.requires_external_software` - Tests needing docking software

### Running by Markers
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only OpenEye tests
pytest tests/ -m "requires_openeye"

# Skip tests requiring external software
pytest tests/ -m "not requires_external_software"
```

## Troubleshooting

### Common Issues

1. **Environment Error**: `Wrong conda environment`
   ```bash
   conda activate cheminf_3_11
   ```

2. **Missing External Software**: Tests skip automatically if software unavailable
   ```bash
   # Check if software is available
   which gnina
   echo $RBT_ROOT
   echo $PLANTS_HOME
   ```

3. **OpenEye License**: Tests skip if OpenEye not available
   ```python
   # Check OpenEye availability
   try:
       from openeye import oechem
       print("OpenEye available")
   except ImportError:
       print("OpenEye not available")
   ```

4. **Test Data Missing**: Ensure example files exist
   ```bash
   ls -la examples/
   ```

### Verbose Output
```bash
# Get detailed test output
pytest tests/ -v -s --tb=long

# Show print statements during tests
pytest tests/ -s
```

## Manual Inspection

After running tests with the `-s` flag, you'll see nicely formatted output with exact paths to inspect:

```
🧪 RxDock Pipeline Test COMPLETED
📁 Output Directory: /tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/
📋 Copy-paste path: /tmp/pytest-of-user/pytest-current/test_rxdock_pipeline_execution0/
🔍 Key files to inspect:
   - Output files: /path/to/output/*.sd
   - RxDock params: /path/to/rxdock.prm
```

**Simply copy the path after "📋 Copy-paste path:" and explore!**

### What to inspect manually:
- ✅ Verify molecular structures in SDF files
- ✅ Check docking poses are reasonable  
- ✅ Validate binding site placement
- ✅ Review score distributions
- ✅ Confirm no obvious errors in outputs

The temporary directories contain complete pipeline outputs exactly as they would be generated in production use.

## Extending Tests

To add new tests:

1. **Add to existing test classes** for related functionality
2. **Use existing fixtures** for consistent setup
3. **Follow the validation pattern**: file existence + non-empty checks
4. **Add appropriate markers** for test categorization
5. **Use `persistent_tmp_workdir`** for outputs you want to inspect manually

### Example New Test
```python
@pytest.mark.slow
def test_my_new_feature(
    persistent_tmp_workdir,
    protein_pdb,
    crystal_sdf,
    ligands_sdf,
    toolkit,
    small_test_settings,
    output_validator
):
    # Your test implementation
    # Use output_validator functions for consistent validation
    pass
```