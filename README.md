# ComplexDocking: Molecular Docking Pipeline

A flexible Python package for molecular docking using multiple docking engines (PLANTS, GNINA, RxDock, and OpenEye).

## Features

- Support for multiple docking engines (PLANTS, GNINA, RxDock, OpenEye) in a single unified interface
- Flexible input formats (SDF or SMILES)
- Automatic stereoisomer enumeration
- Choice of molecule preparation toolkit (CDPKit or OpenEye)
- Choice of protein protonation method (PDBFixer or Protoss)
- Collection of results in pandas DataFrames
- Post-docking analysis with RMSD-based clustering (DBSCAN and K-medoids)
- Protein-ligand complex minimization using OpenMM
- Automatic extraction of both GNINA and SMINA scoring functions

## Requirements

- Python 3.8+
- RDKit
- Datamol
- Pandas
- Biotite
- OpenBabel/PyBel
- CDPKit (for ligand preparation)
- OpenEye toolkit (optional, for alternative ligand preparation)
- Protoss (default for protein protonation)
- PDBFixer (alternative for protein protonation)
- PLANTS docking software
- GNINA docking software
- OpenEye docking software (optional)
- OpenMM (for protein-ligand complex minimization)
- Scikit-learn (for clustering)
- PyMOL (for visualization, optional)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ComplexDocking.git
cd ComplexDocking

# Install dependencies
conda env create -f environment.yml
conda activate complex-docking
```

## Usage

### Basic Usage

```python
from pathlib import Path
from Wrapper_Docking import PipelineDocking

# Initialize the docking pipeline
docking = PipelineDocking(
    workdir=Path("./Docking"),
    docking_software=["plants", "gnina"],  # Choose one or more: "plants", "gnina", "rxdock", "openeye"
    settings=(10, 4),  # (n_conformers, n_cpus)
    protein_pdb=Path("./0_Examples/8gcy.pdb"),
    ligands_input=Path("./0_Examples/some_ligands.sdf"),  # Can be SDF or SMILES file
    crystal_sdf=Path("./0_Examples/Crystal.sdf"),
    toolkit="cdpkit",  # Choose "cdpkit" or "openeye"
    protonation_method="protoss"  # Choose "protoss" or "pdbfixer"
)

# Run the docking and get results
results = docking.run()

# Access results for each docking software
plants_results = results.get("plants")
gnina_results = results.get("gnina")
smina_results = results.get("smina")  # SMINA scoring is automatically extracted from GNINA results
```

### Using SMILES Input

```python
from pathlib import Path
from Wrapper_Docking import PipelineDocking

# With SMILES input (CSV file with SMILES column)
docking = PipelineDocking(
    workdir=Path("./Docking"),
    docking_software=["plants", "gnina"],
    settings=(10, 4),
    protein_pdb=Path("./0_Examples/8gcy.pdb"),
    ligands_input=Path("./0_Examples/compounds.csv"),  # CSV with 'SMILES' column
    crystal_sdf=Path("./0_Examples/Crystal.sdf"),
    toolkit="cdpkit",
    protonation_method="protoss"
)

results = docking.run()
```

### Using OpenEye Toolkit

```python
from pathlib import Path
from Wrapper_Docking import PipelineDocking

# Using OpenEye toolkit for ligand preparation
docking = PipelineDocking(
    workdir=Path("./Docking"),
    docking_software=["plants", "gnina", "rxdock", "openeye"],
    settings=(10, 4),
    protein_pdb=Path("./0_Examples/8gcy.pdb"),
    ligands_input=Path("./0_Examples/some_ligands.sdf"),
    crystal_sdf=Path("./0_Examples/Crystal.sdf"),
    toolkit="openeye",  # Use OpenEye toolkit (if available)
    protonation_method="protoss"
)

results = docking.run()
```

## Post-Docking Analysis

### Clustering of Docking Poses

The package provides tools for clustering docking poses based on RMSD calculations, which can help identify distinct binding modes and select diverse representative structures.

#### Using DBSCAN Clustering

```python
from functools import partial
from HandsFreeDocking.analysis.clustering import (
    OptimizedDBSCANClustering, PairwiseMatrixComputer, 
    calc_rmsd_mcs_with_timeout
)

# Assuming FULL_DF is a DataFrame with docking results containing a "Molecule" column
molecules = FULL_DF["Molecule"].tolist()

# Calculate pairwise RMSD matrix
computer = PairwiseMatrixComputer(molecules, n_jobs=8, timeout=60)
rmsd_funct = partial(calc_rmsd_mcs_with_timeout, timeout=60)
rmsd_matrix = computer.compute_matrix(rmsd_funct)

# Perform DBSCAN clustering with automatic parameter optimization
clustering_dbscan = OptimizedDBSCANClustering(
    eps_range=(0.5, 5.0, 0.5),        # Range of epsilon values to try
    min_samples_range=(2, 15),        # Range of min_samples values to try
    max_noise_percent=15.0,           # Maximum allowed percentage of noise points
    max_clusters=10,                  # Maximum number of clusters desired
    use_dimensionality_reduction=True,# Use dimensionality reduction for better visualization
    verbose=True                      # Show progress and results
)

# Fit the model and get cluster labels
labels_dbscan = clustering_dbscan.fit(rmsd_matrix)
results_dbscan = clustering_dbscan.get_results()

# Add cluster labels to the dataframe
FULL_DF["Cluster_DBSCAN"] = labels_dbscan
```

#### Using K-Medoids Clustering

```python
from HandsFreeDocking.analysis.clustering import OptimizedKMedoidsClustering

# Perform K-medoids clustering with automatic parameter optimization
clustering_kmed = OptimizedKMedoidsClustering(
    k_range=(2, 20),                  # Range of k values to try
    use_dimensionality_reduction=True,# Use dimensionality reduction for better visualization
    verbose=True                      # Show progress and results
)

# Fit the model and get cluster labels
labels_kmed = clustering_kmed.fit(rmsd_matrix)
results_kmed = clustering_kmed.get_results()

# Add cluster labels to the dataframe
FULL_DF["Cluster_KMedoids"] = labels_kmed
```

### Visualizing Clusters in PyMOL

```python
from pymol import cmd
from tempfile import NamedTemporaryFile
import datamol as dm
from rdkit import Chem

# Load protein
cmd.reinitialize()
cmd.load(protein_pdb_path)

# Helper function to save molecules to temporary files
def tmp_save(mol: Chem.rdchem.Mol):
    with NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        dm.to_sdf(mol, f.name)
        return f.name

# Group compounds by cluster and select top scoring poses
for cluster, df in FULL_DF.groupby("Cluster_DBSCAN"):
    IDs = []
    # For each ligand in the cluster, get the top scoring pose
    for ndx, sub_df in df.groupby("Lig_Name"):
        sub_df.sort_values(by="Score", ascending=False, inplace=True)
        TOP = sub_df.iloc[0]
        
        ID = TOP["ID"]
        IDs.append(ID)
        TMP_MOL = tmp_save(TOP["Molecule"])
    
        cmd.load(TMP_MOL, ID)
    
    # Group all molecules in the same cluster
    cmd.group(f"Cluster_{cluster}", " ".join(IDs))

# Save the PyMOL session
cmd.save("clustered_poses.pse")
```

### Protein-Ligand Complex Minimization

The package provides functionality to minimize protein-ligand complexes using OpenMM, which can help refine docking poses and improve binding energy calculations.

```python
from HandsFreeDocking.tools.Protein_Minimization import ProteinMinimizer

# Initialize the minimizer with a DataFrame containing docking results
# - docked_mol_col: column containing RDKit molecules
# - pdb_path_col: column containing paths to PDB files
protein_minimizer = ProteinMinimizer(FULL_DF, "Molecule", "Protein_Path")

# Run minimization on all complexes
# This will add new columns to the DataFrame:
# - PDB_Min: minimized PDB structure
# - Delta_RMSD: RMSD between original and minimized pose
# - Delta_Energy: energy difference after minimization
# - ERROR: any errors encountered during minimization
minimized_df = protein_minimizer()

# Access minimization results
for idx, row in minimized_df.iterrows():
    print(f"Molecule {row['ID']}:")
    print(f"  RMSD: {row['Delta_RMSD']:.2f} Å")
    print(f"  Energy Change: {row['Delta_Energy']:.2f} kJ/mol")
    
    # Save minimized structure if needed
    if row['PDB_Min'] is not None:
        with open(f"{row['ID']}_minimized.pdb", "w") as f:
            f.write(row['PDB_Min'])
```

## Individual Docking Engines

### PLANTS Docking

```python
from pathlib import Path
from src.Plants_Pipeline import Plants_Docking

# Initialize and run Plants docking
plants_docking = Plants_Docking(
    workdir=Path("./Docking"),
    pdb_ID=Path("./0_Examples/8gcy.pdb"),
    crystal_path=Path("./0_Examples/Crystal.sdf"),
    ligands_sdf=Path("./0_Examples/some_ligands.sdf"),
    toolkit="cdpkit"  # or "openeye"
)

plants_docking.main(n_confs=10, n_cpus=4)
```

### GNINA Docking

```python
from pathlib import Path
from src.Gnina_Pipeline import Gnina_Docking

# Initialize and run Gnina docking (non-covalent)
gnina_docking = Gnina_Docking(
    workdir=Path("./Docking"),
    pdb_ID=Path("./0_Examples/8gcy.pdb"),
    crystal_path=Path("./0_Examples/Crystal.sdf"),
    ligands_sdf=Path("./0_Examples/some_ligands.sdf"),
    toolkit="cdpkit",  # or "openeye"
    protonation_method="protoss"  # or "pdbfixer"
)

# For non-covalent docking
gnina_docking.non_covalent_run(n_confs=10, n_cpus=4)

# For covalent docking
gnina_docking.covalent_run(
    n_confs=10, 
    n_cpus=4, 
    atom_to_covalent="B:220:SG",  # Receptor atom to form covalent bond
    smarts_react="[#6]=[#6]"      # SMARTS pattern for reactive group
)
```

### OpenEye Docking

```python
from pathlib import Path
from src.OpenEye_Pipeline import OpenEye_Docking

# Prepare ligands as (SMILES, ID) tuples
docking_tuple = [
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic_acid")
]

# Initialize and run OpenEye docking
openeye_docking = OpenEye_Docking(
    workdir=Path("./Docking"),
    pdb_ID=Path("./0_Examples/8gcy.pdb"),
    mtz=None,
    crystal_path=Path("./0_Examples/Crystal.sdf"),
    docking_tuple=docking_tuple
)

openeye_docking.run_oedocking_pipeline(
    n_cpu=4, 
    confs=10, 
    mtz=None, 
    mode="oe"
)
```

## Complete Workflow Example

Here's a complete workflow example including docking, clustering, and minimization:

```python
from pathlib import Path
import pandas as pd
from functools import partial
import pickle

# 1. Docking
from HandsFreeDocking.Wrapper_Docking import PipelineDocking

docking = PipelineDocking(
    workdir=Path("./Docking"),
    docking_software=["plants", "gnina", "openeye"],
    settings=(10, 4),  # (n_conformers, n_cpus)
    protein_pdb=Path("./protein.pdb"),
    ligands_input=Path("./ligands.sdf"),
    crystal_sdf=Path("./crystal.sdf"),
    toolkit="cdpkit"
)

results = docking.run()
FULL_DF = docking.concat_df()

# Save results for later use
with open("docking_results.pkl", "wb") as f:
    pickle.dump(FULL_DF, f)

# 2. Clustering of poses
from HandsFreeDocking.analysis.clustering import (
    OptimizedDBSCANClustering, PairwiseMatrixComputer, 
    calc_rmsd_mcs_with_timeout
)

# Get all molecules
ALL_MOLS = FULL_DF["Molecule"].tolist()

# Calculate pairwise RMSD matrix
computer = PairwiseMatrixComputer(ALL_MOLS, n_jobs=8, timeout=60)
rmsd_funct = partial(calc_rmsd_mcs_with_timeout, timeout=60)
rmsd_matrix = computer.compute_matrix(rmsd_funct)

# DBSCAN clustering
clustering_dbscan = OptimizedDBSCANClustering(
    eps_range=(0.5, 5.0, 0.5),
    min_samples_range=(2, 15),
    max_noise_percent=15.0,
    max_clusters=10,
    use_dimensionality_reduction=True,
    verbose=True
)
labels_dbscan = clustering_dbscan.fit(rmsd_matrix)
FULL_DF["Cluster_DBSCAN"] = labels_dbscan

# 3. Protein-ligand complex minimization
from HandsFreeDocking.tools.Protein_Minimization import ProteinMinimizer

protein_minimizer = ProteinMinimizer(FULL_DF, "Molecule", "Protein_Path")
minimized_df = protein_minimizer()

# 4. Analysis and visualization
# Group by cluster and extract top scoring poses
cluster_representatives = []

for cluster, df in minimized_df.groupby("Cluster_DBSCAN"):
    # Skip noise (cluster = -1)
    if cluster == -1:
        continue
        
    for lig_name, sub_df in df.groupby("Lig_Name"):
        # Sort by score (higher is better)
        sub_df = sub_df.sort_values(by="Score", ascending=False)
        
        # Take the top pose
        top_pose = sub_df.iloc[0]
        
        cluster_representatives.append({
            "ID": top_pose["ID"],
            "Lig_Name": lig_name,
            "Cluster": cluster,
            "Score": top_pose["Score"],
            "Delta_Energy": top_pose["Delta_Energy"],
            "Delta_RMSD": top_pose["Delta_RMSD"]
        })

# Create a summary DataFrame
summary_df = pd.DataFrame(cluster_representatives)
summary_df.to_csv("cluster_representatives.csv", index=False)

print("Complete workflow finished. Results saved.")
```

## File Structure

```
ComplexDocking/
├── Wrapper_Docking.py       # Main wrapper class
├── HandsFreeDocking/
│   ├── Plants_Pipeline.py   # PLANTS docking implementation
│   ├── Gnina_Pipeline.py    # GNINA docking implementation 
│   ├── RxDock_Pipeline.py   # RxDock docking implementation
│   ├── OpenEye_Pipeline.py  # OpenEye docking implementation
│   ├── tools/               # Utility functions and helper classes
│   │   ├── Protein_Preparation.py  # Protein preparation using Protoss
│   │   └── Protein_Minimization.py # Protein-ligand complex minimization
│   └── analysis/
│       └── clustering.py    # Clustering algorithms for pose analysis
├── 0_Examples/              # Example input files
│   ├── 8gcy.pdb             # Example protein structure
│   ├── Crystal.sdf          # Example crystal ligand
│   ├── some_ligands.sdf     # Example ligands for docking
│   └── compounds.csv        # Example SMILES input
└── README.md
```

## Notes

- PLANTS requires a special environment variable which is set automatically in the Plants_Pipeline.py
- GNINA should be available in your PATH for Gnina_Pipeline.py to work
- RxDock requires setting RBT_ROOT environment variable which is handled automatically in RxDock_Pipeline.py
- OpenEye requires a valid license to use the OpenEye toolkit and docking software
- Protoss is the default protein protonation method; PDBFixer is available as an alternative
- The binding site is defined based on the crystal ligand for all docking methods
- All docking methods use internal multiprocessing, so the wrapper runs them sequentially
- Protein minimization requires OpenMM and is computationally intensive; CUDA is recommended for better performance
- SMINA scoring is automatically extracted when using GNINA docking
- The recommended workflow is: docking → clustering → minimization
