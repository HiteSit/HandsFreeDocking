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