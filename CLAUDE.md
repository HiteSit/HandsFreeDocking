# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HandsFreeDocking is a flexible Python package for molecular docking using multiple docking engines (PLANTS, GNINA, RxDock, and OpenEye) in a unified interface. The package handles ligand preparation, protein protonation, docking execution, and post-docking analysis including clustering and minimization.

## Key Development Commands

### Environment Setup
```bash
conda activate cheminf_3_11
```

### Running Tests
```bash
pytest test/
```

### Running Docking Pipelines
The main entry point is through `HandsFreeDocking.Wrapper_Docking.PipelineDocking`. Individual docking engines can also be accessed directly through their respective pipeline files.

## Architecture Overview

### Core Components

1. **Main Wrapper** (`HandsFreeDocking/Wrapper_Docking.py`)
   - `PipelineDocking` class orchestrates multiple docking engines
   - Handles input validation, preparation, and result aggregation
   - Supports both SDF and SMILES inputs
   - Manages protein preparation (Protoss/PDBFixer) and ligand preparation (CDPKit/OpenEye)

2. **Docking Engine Pipelines** (in `HandsFreeDocking/`)
   - `Gnina_Pipeline.py`: GNINA docking (includes SMINA scoring extraction)
   - `Plants_Pipeline.py`: PLANTS docking
   - `RxDock_Pipeline.py`: RxDock docking  
   - `OpenEye_Pipeline.py`: OpenEye OMEGA/OEDocking
   - Each pipeline follows a similar pattern: initialization → preparation → docking → result collection

3. **Tools** (`HandsFreeDocking/tools/`)
   - `CDPK_Utils.py`: CDPKit-based ligand preparation utilities
   - `OpeneEye_Utils.py`: OpenEye toolkit utilities
   - `Protein_Preparation.py`: Protein protonation using Protoss
   - `Protein_Minimization.py`: OpenMM-based complex minimization
   - `tools.py`: Common utilities for molecule handling

4. **Analysis** (`HandsFreeDocking/analysis/`)
   - `clustering.py`: RMSD-based clustering (DBSCAN, K-medoids)
   - `clustering_GPU.py`: GPU-accelerated clustering (if available)

5. **HPC Support** (`HPC/`)
   - `Wrapper_MPI.py`: MPI wrapper for distributed computing
   - `Wrapper_MPI_Run.sh`: SLURM submission script

### Key Design Patterns

- **Pipeline Pattern**: Each docking engine implements a similar pipeline interface
- **Factory Pattern**: Toolkit selection (CDPKit vs OpenEye) is handled via string parameter
- **DataFrame-centric**: Results are collected in pandas DataFrames for easy analysis
- **Multiprocessing**: Built-in support for parallel execution within each engine

### Important Implementation Details

- Docking engines require external software (PLANTS, GNINA, RxDock, OpenEye) to be installed separately
- Binding sites are automatically defined based on crystal ligand structures
- Stereoisomer enumeration is handled automatically during preparation
- Each docking engine runs sequentially when multiple are selected (internal parallelization)
- Results include normalized scores for cross-engine comparison

### External Dependencies

The package relies on several external tools that must be installed:
- PLANTS: Requires PLANTS_HOME environment variable
- GNINA: Must be in PATH
- RxDock: Requires RBT_ROOT environment variable
- OpenEye: Requires valid license
- Protoss: Default protein preparation tool

### Development Reminders

- When running any script or Python-related task, **always ensure that the `cheminf_3_11` environment is activated**

## Recent Changes and Refactoring

### Tautomer Enumeration Refactoring (Session: January 2025)

#### Problem Identified
The original `LigandPreparator` class had a conceptually incorrect design with two separate methods (`_get_best_tautomer` and `_get_best_tautomers`) and a confusing `enumerate_tautomers` boolean parameter. This didn't match the elegant single-parameter design from the original notebook prototype.

#### Changes Made

**1. Core `LigandPreparator` Class (`HandsFreeDocking/tools/Ligand_Preparation.py`)**
- **Removed**: `enumerate_tautomers: bool` parameter from `__init__`
- **Unified**: Single `_get_best_tautomer` method that behaves like the original notebook function:
  - `tautomer_score_threshold=None` → returns single best tautomer (`Chem.Mol`)
  - `tautomer_score_threshold=value` → returns list of tautomers within threshold (`List[Chem.Mol]`)
- **Removed**: `_get_best_tautomers` method entirely
- **Updated**: `_process_single_molecule` to handle both single molecule and list returns
- **Fixed**: Naming convention to properly create `{base_name}_Iso{i}_Taut{j}` for combinatorial explosion

**2. Pipeline Classes Updated**
All docking pipeline classes now accept `tautomer_score_threshold: Optional[float] = None`:
- `RxDock_Pipeline.py`: Added parameter, removed `enumerate_tautomers=False` usage
- `Plants_Pipeline.py`: Added parameter, removed `enumerate_tautomers=False` usage  
- `Gnina_Pipeline.py`: Added parameter, removed `enumerate_tautomers=False` usage
- **Design principle**: No defaults in pipeline classes - they receive values from wrapper

**3. Wrapper Integration (`HandsFreeDocking/Wrapper_Docking.py`)**
- **Added**: `tautomer_score_threshold: Optional[float] = None` parameter (ONLY place with default)
- **Updated**: All pipeline instantiations to pass the parameter
- **Updated**: `_process_input_to_sdf` method to pass parameter to `LigandPreparator`

**4. Test Updates**
- **RxDock tests**: Use `tautomer_score_threshold=2.0` (enables multiple tautomers per stereoisomer)
- **Plants/Gnina tests**: Use `tautomer_score_threshold=None` (default: best tautomer only)
- **Integration tests**: Updated to pass the new parameter

**5. Import Fixes**
- **Added**: `Optional` type hints to all necessary files
- **Updated**: Import statements in pipeline and wrapper files

#### Naming Convention Results
- `tautomer_score_threshold=None`: `Ligand_A_Iso0`, `Ligand_A_Iso1` (no Taut suffix)
- `tautomer_score_threshold=2.0`: `Ligand_A_Iso0_Taut0`, `Ligand_A_Iso0_Taut1`, `Ligand_A_Iso1_Taut0` (combinatorial explosion)

#### Verification
- **Tests passed**: RxDock pipeline tests completed successfully
- **Backward compatibility**: Maintained for existing workflows
- **Parameter exposure**: Tautomer control now available at wrapper level for easy tuning

#### Key Benefits
1. **Unified Interface**: Single method matches original notebook design
2. **Correct Combinatorial Logic**: Proper stereoisomer → tautomer enumeration flow  
3. **Flexible Control**: Parameter exposed at wrapper level for easy configuration
4. **Consistent Naming**: Clear `_Iso{i}_Taut{j}` pattern for tracking molecule variants
5. **Maintains Defaults**: Existing pipelines continue to work with sensible defaults