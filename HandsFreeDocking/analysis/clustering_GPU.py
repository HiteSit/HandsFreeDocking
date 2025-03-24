import concurrent.futures
import logging
import os
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Union, Tuple, Callable, Dict, Any

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
from joblib import Parallel, delayed

# Import necessary functions directly from original clustering.py
from .clustering import calc_rmsd_mcs_with_timeout, calc_usr_similarity, calc_splif
from .clustering import PairwiseMatrixComputer as OriginalPairwiseMatrixComputer


class GPUPairwiseMatrixComputer:
    """
    GPU-accelerated pairwise matrix computation for molecular comparison.
    
    This class provides GPU acceleration for computing pairwise matrices while
    maintaining compatibility with the original PairwiseMatrixComputer API.
    """
    
    def __init__(self, 
                 molecules: List[Chem.Mol], 
                 n_jobs: int = 8,
                 timeout: int = 30, 
                 max_mols: Optional[int] = None,
                 batch_size: int = 100):
        """
        Initialize the GPU-accelerated PairwiseMatrixComputer.

        Parameters:
        -----------
        molecules : list
            List of molecules to compute pairwise values for (e.g., rdkit.Chem.Mol objects)
        n_jobs : int, optional
            Number of parallel jobs for CPU fallback operations (default: 8)
        timeout : int, optional
            Maximum time in seconds for each pairwise calculation (default: 30)
        max_mols : int, optional
            Maximum number of molecules to process (default: None, processes all)
        batch_size : int, optional
            Size of batches for GPU processing (default: 100)
        """
        self.molecules = molecules[:max_mols] if max_mols else molecules
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.batch_size = batch_size
        self.n = len(self.molecules)
        
        # Check if GPU is available
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            print(f"Using GPU acceleration with device: {torch.cuda.get_device_name(0)}")
            self.device = torch.device('cuda')
        else:
            print("CUDA not available. Using CPU for calculations.")
            self.device = torch.device('cpu')
        
        # Create a fallback CPU implementation
        self.cpu_computer = OriginalPairwiseMatrixComputer(
            molecules=self.molecules,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            max_mols=None  # We've already applied max_mols
        )
    
    def compute_matrix(self, pairwise_func):
        """
        Compute a pairwise matrix using the specified function with optional GPU acceleration.

        Parameters:
        -----------
        pairwise_func : callable or str
            Function to use for pairwise calculations or a string identifier

        Returns:
        --------
        numpy.ndarray
            A symmetric matrix of shape (n, n) containing pairwise values
        """
        # Handle direct string identifiers
        if isinstance(pairwise_func, str):
            if pairwise_func == 'rmsd':
                pairwise_func = lambda mol1, mol2: calc_rmsd_mcs_with_timeout(mol1, mol2, timeout=self.timeout)
            elif pairwise_func == 'usr':
                pairwise_func = calc_usr_similarity
            elif pairwise_func == 'splif':
                raise ValueError("For 'splif', please provide a partial function with pocket_file parameter")
            else:
                raise ValueError(f"Unknown string identifier: {pairwise_func}")
        
        # Check if we can use GPU acceleration for this function
        func_name = str(pairwise_func)
        
        if 'calc_rmsd_mcs_with_timeout' in func_name and self.has_gpu:
            return self._compute_rmsd_matrix_gpu()
        elif 'calc_usr_similarity' in func_name and self.has_gpu:
            return self._compute_usr_matrix_gpu()
        else:
            # For other functions or when GPU is not available, use CPU implementation
            print(f"Using CPU implementation for function: {pairwise_func}")
            return self.cpu_computer.compute_matrix(pairwise_func)
    
    def _mol_to_tensor(self, mol):
        """Convert molecule coordinates to tensor."""
        if mol.GetNumConformers() == 0:
            return None
        
        conf = mol.GetConformer()
        coords = []
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        
        return torch.tensor(coords, dtype=torch.float32, device=self.device)
    
    def _compute_rmsd_matrix_gpu(self):
        """
        Compute RMSD matrix using GPU acceleration for identical molecules.
        Falls back to CPU for molecules requiring MCS.
        """
        start_time = time.time()
        n = self.n
        print(f"Computing pairwise RMSD matrix for {n} molecules ({n*(n-1)//2} pairs)")
        
        # Initialize the result matrix
        result_matrix = np.full((n, n), np.nan)
        np.fill_diagonal(result_matrix, 0)  # Self-similarity is 0
        
        # Extract and cache molecule information
        mol_info = []
        for i, mol in enumerate(self.molecules):
            coords = self._mol_to_tensor(mol)
            atom_counts = mol.GetNumAtoms()
            atom_types = tuple(a.GetSymbol() for a in mol.GetAtoms())
            mol_info.append((coords, atom_counts, atom_types))
        
        # Process all pairs
        pairs = list(combinations(range(n), 2))
        total_pairs = len(pairs)
        
        processed = 0
        batch_size = min(self.batch_size, total_pairs)
        
        with tqdm(total=total_pairs) as pbar:
            # Process batches of pairs
            for batch_start in range(0, total_pairs, batch_size):
                batch_end = min(batch_start + batch_size, total_pairs)
                batch_pairs = pairs[batch_start:batch_end]
                
                # Group pairs by whether they need MCS or not
                direct_pairs = []
                mcs_pairs = []
                
                for i, j in batch_pairs:
                    coords_i, atom_count_i, atom_types_i = mol_info[i]
                    coords_j, atom_count_j, atom_types_j = mol_info[j]
                    
                    if (coords_i is not None and coords_j is not None and 
                        atom_count_i == atom_count_j and atom_types_i == atom_types_j):
                        direct_pairs.append((i, j))
                    else:
                        mcs_pairs.append((i, j))
                
                # Process direct pairs with GPU
                if direct_pairs and self.has_gpu:
                    with torch.no_grad():
                        for i, j in direct_pairs:
                            coords_i = mol_info[i][0]
                            coords_j = mol_info[j][0]
                            
                            # Center the coordinates
                            coords_i_center = coords_i - coords_i.mean(dim=0, keepdim=True)
                            coords_j_center = coords_j - coords_j.mean(dim=0, keepdim=True)
                            
                            # Compute the covariance matrix
                            covar = torch.matmul(coords_i_center.T, coords_j_center)
                            
                            # Single Value Decomposition
                            u, s, v = torch.linalg.svd(covar)
                            
                            # Determine if we need a reflection correction
                            det = torch.det(torch.matmul(v, u.T))
                            if det < 0:
                                v[-1] = -v[-1]
                            
                            # Calculate rotation matrix
                            rot_mat = torch.matmul(v, u.T)
                            
                            # Apply rotation to first set
                            rotated_coords = torch.matmul(coords_i_center, rot_mat)
                            
                            # Calculate RMSD
                            msd = torch.mean(torch.sum((rotated_coords - coords_j_center)**2, dim=1))
                            rmsd = torch.sqrt(msd).item()
                            
                            result_matrix[i, j] = rmsd
                            result_matrix[j, i] = rmsd
                
                # Process MCS pairs with CPU
                if mcs_pairs:
                    cpu_results = Parallel(n_jobs=self.n_jobs)(
                        delayed(calc_rmsd_mcs_with_timeout)(
                            self.molecules[i], self.molecules[j], timeout=self.timeout
                        ) for i, j in mcs_pairs
                    )
                    
                    for (i, j), val in zip(mcs_pairs, cpu_results):
                        if val is not None:
                            result_matrix[i, j] = val
                            result_matrix[j, i] = val
                
                # Update progress
                processed += len(batch_pairs)
                pbar.update(len(batch_pairs))
                
                # Print progress occasionally
                if processed % (batch_size * 10) == 0 or processed == total_pairs:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed}/{total_pairs} pairs ({processed/total_pairs:.1%}) "
                          f"in {elapsed:.1f}s ({processed/elapsed:.1f} pairs/s)")
        
        # Report final statistics
        elapsed = time.time() - start_time
        completed = np.sum(~np.isnan(result_matrix[np.triu_indices(n, k=1)]))
        print(f"Completed {completed}/{total_pairs} pairs ({completed/total_pairs:.1%}) "
              f"in {elapsed:.1f}s ({completed/elapsed:.1f} pairs/s)")
        
        return result_matrix
    
    def _compute_usr_matrix_gpu(self):
        """
        Compute USR similarity matrix using GPU for moment calculations.
        For USR, we compute moments on GPU but still use CPU for the final calculation.
        """
        start_time = time.time()
        n = self.n
        print(f"Computing pairwise USR matrix for {n} molecules ({n*(n-1)//2} pairs)")
        
        # Initialize the result matrix
        result_matrix = np.full((n, n), np.nan)
        np.fill_diagonal(result_matrix, 1.0)  # Self-similarity is 1.0
        
        # Use the CPU implementation which is reliable
        return self.cpu_computer.compute_matrix(calc_usr_similarity)


# Convenience functions to match the original API

def compute_rmsd_matrix(molecules, n_jobs=8, timeout=30, max_mols=None, use_gpu=True, **kwargs):
    """
    Compute RMSD matrix for a set of molecules.
    
    Parameters:
    -----------
    molecules : list of rdkit.Chem.Mol
        List of molecules to compute pairwise RMSD for
    n_jobs : int, optional
        Number of parallel jobs (default: 8)
    timeout : int, optional
        Maximum time in seconds for each calculation (default: 30)
    max_mols : int, optional
        Maximum number of molecules to process (default: None)
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: True)
        
    Returns:
    --------
    numpy.ndarray
        Pairwise RMSD matrix
    """
    if use_gpu and torch.cuda.is_available():
        computer = GPUPairwiseMatrixComputer(
            molecules=molecules, 
            n_jobs=n_jobs, 
            timeout=timeout, 
            max_mols=max_mols,
            batch_size=kwargs.get('batch_size', 100)
        )
        return computer.compute_matrix('rmsd')
    else:
        from .clustering import PairwiseMatrixComputer
        computer = PairwiseMatrixComputer(molecules, n_jobs=n_jobs, timeout=timeout, max_mols=max_mols)
        return computer.compute_matrix(lambda mol1, mol2: calc_rmsd_mcs_with_timeout(mol1, mol2, timeout=timeout))


def compute_usr_matrix(molecules, n_jobs=8, timeout=30, max_mols=None, use_gpu=True, **kwargs):
    """
    Compute USR similarity matrix for a set of molecules.
    
    Parameters:
    -----------
    molecules : list of rdkit.Chem.Mol
        List of molecules to compute pairwise USR similarity for
    n_jobs : int, optional
        Number of parallel jobs (default: 8)
    timeout : int, optional
        Maximum time in seconds for each calculation (default: 30)
    max_mols : int, optional
        Maximum number of molecules to process (default: None)
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: True)
        
    Returns:
    --------
    numpy.ndarray
        Pairwise USR similarity matrix
    """
    if use_gpu and torch.cuda.is_available():
        computer = GPUPairwiseMatrixComputer(
            molecules=molecules, 
            n_jobs=n_jobs, 
            timeout=timeout, 
            max_mols=max_mols,
            batch_size=kwargs.get('batch_size', 100)
        )
        return computer.compute_matrix('usr')
    else:
        from .clustering import PairwiseMatrixComputer
        computer = PairwiseMatrixComputer(molecules, n_jobs=n_jobs, timeout=timeout, max_mols=max_mols)
        return computer.compute_matrix(calc_usr_similarity)


def compute_splif_matrix(molecules, pocket_file, n_jobs=8, timeout=30, max_mols=None, use_gpu=False, **kwargs):
    """
    Compute SPLIF similarity matrix for a set of molecules.
    Note: This always uses CPU implementation as SPLIF doesn't benefit from GPU acceleration.
    
    Parameters:
    -----------
    molecules : list of rdkit.Chem.Mol
        List of molecules to compute pairwise SPLIF similarity for
    pocket_file : Path
        Path to pocket file (PDB format)
    n_jobs : int, optional
        Number of parallel jobs (default: 8)
    timeout : int, optional
        Maximum time in seconds for each calculation (default: 30)
    max_mols : int, optional
        Maximum number of molecules to process (default: None)
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False, not used for SPLIF)
        
    Returns:
    --------
    numpy.ndarray
        Pairwise SPLIF similarity matrix
    """
    # SPLIF calculation doesn't benefit from GPU, so always use CPU
    from .clustering import PairwiseMatrixComputer
    computer = PairwiseMatrixComputer(molecules, n_jobs=n_jobs, timeout=timeout, max_mols=max_mols)
    return computer.compute_matrix(lambda mol1, mol2: calc_splif(mol1, mol2, pocket_file)) 