from pathlib import Path
from typing import List, Tuple, Optional

import os
import sys
import traceback

import numpy as np
from mpi4py import MPI
from rdkit import Chem

from HandsFreeDocking.Gnina_Pipeline import Gnina_Docking

def get_sub_workdir(main_dir: Path, rank: int) -> Path:
    """Generate the subdirectory path for a given rank."""
    return main_dir / f"Rank_{rank}"

def split_sdf_for_mpi(ligands_sdf: Path, main_dir: Path, size: int):
    """
    Split molecules from an SDF file into multiple SDF files for MPI processes.
    
    Args:
        ligands_sdf (Path): Path to the main SDF file.
        main_dir (Path): Directory where subdirectories and SDF files will be created.
        size (int): Number of MPI processes.
    """
    supplier = Chem.SDMolSupplier(str(ligands_sdf))
    nMols = len(supplier)
    
    n_sub_mols = nMols // size
    n_sub_mols_left = nMols % size
    
    supplier = Chem.SDMolSupplier(str(ligands_sdf))
    
    for i in range(size):
        n_mols_i = n_sub_mols + 1 if i < n_sub_mols_left else n_sub_mols
        sub_workdir = get_sub_workdir(main_dir, i)
        sub_ligands_sdf = sub_workdir / "ligands.sdf"
        
        with Chem.SDWriter(str(sub_ligands_sdf)) as writer:
            for _ in range(n_mols_i):
                try:
                    mol = next(supplier)
                    if mol is not None:
                        writer.write(mol)
                except StopIteration:
                    print(f"Warning: Reached end of supplier at process {i}", flush=True)
                    break

def mpi_runner(
        sub_workdir: Path,
        pdb_path: Path,
        crystal_path: Path,
        sub_ligands_sdf: Path
    ):
    
    gnina_docking = Gnina_Docking(
        workdir=sub_workdir,
        pdb_ID=pdb_path,
        crystal_path=crystal_path,
        ligands_sdf=sub_ligands_sdf,
        protonation_method="oe",
        protein_protonation_method="protoss"
    )

    gnina_docking.covalent_run(
        n_confs=10, 
        n_cpus=1, 
        atom_to_covalent="D:220:SG",
        smarts_react="[#6]=[#6]"
    )

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    main_dir = Path("./RUN_2_p53/Docking_Results")
    pdb_path = Path("./RUN_2_p53/8dc4_D.pdb")
    crystal_path = Path("./RUN_2_p53/Crystal.sdf")
    ligands_sdf = Path("./RUN_2_p53/Imma_VS.sdf")

    if rank == 0:
        main_dir.mkdir(exist_ok=True)
        [get_sub_workdir(main_dir, i).mkdir(exist_ok=True) for i in range(size)]
        split_sdf_for_mpi(ligands_sdf, main_dir, size)

    comm.Barrier()

    sub_workdir = get_sub_workdir(main_dir, rank)
    sub_ligands_sdf = sub_workdir / "ligands.sdf"
    
    # Proceed with mpi_runner or other operations
    mpi_runner(sub_workdir, pdb_path, crystal_path, sub_ligands_sdf)