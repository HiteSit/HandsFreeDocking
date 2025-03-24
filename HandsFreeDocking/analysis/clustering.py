import concurrent.futures
import logging
import os
import shutil
import time
import warnings
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import cloudpickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import oddt
import oddt.fingerprints
import oddt.shape
import oddt.toolkits.rdk
import pandas as pd
import tabulate
from espsim import GetEspSim
from IPython.display import clear_output
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFMCS
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn_extra.cluster import KMedoids
from spyrmsd import molecule, rmsd
from tqdm import tqdm

def calc_rmsd_mcs_with_timeout(mol, ref, chirality=True, timeout=30):
    """
    Calculate RMSD between molecules with a timeout mechanism.
    
    This function computes the Root Mean Square Deviation (RMSD) between two molecules
    using either substructure matching or Maximum Common Substructure (MCS) approach.
    The calculation is wrapped in a timeout mechanism to prevent infinite loops.
    
    Parameters:
    -----------
    mol : rdkit.Chem.Mol
        The molecule to compare against the reference
    ref : rdkit.Chem.Mol
        The reference molecule
    chirality : bool, optional
        Whether to consider chirality in the comparison (default: True)
    timeout : int, optional
        Maximum time in seconds to wait for the calculation (default: 30)
        
    Returns:
    --------
    float or None
        The calculated RMSD value rounded to 3 decimal places, or None if:
        - The calculation times out
        - An error occurs during calculation
        - No valid MCS is found between the molecules
    """
    def worker():
        def get_coord(mol, indices=None):
            if indices is None:
                indices = tuple(range(mol.GetNumAtoms()))
            output = []
            conformer = mol.GetConformer()
            for atom_id in indices:
                pos = conformer.GetAtomPosition(atom_id)
                output.append((pos.x, pos.y, pos.z))
            return tuple(output)

        def rmsd_calc(r_coord, m_coord):
            s = 0
            for r, m in zip(r_coord, m_coord):
                s += (r[0] - m[0]) ** 2 + (r[1] - m[1]) ** 2 + (r[2] - m[2]) ** 2
            s = (s / len(r_coord)) ** 0.5
            return s

        try:
            match_indices = mol.GetSubstructMatches(ref, uniquify=False, useChirality=chirality, maxMatches=10000)
            min_rmsd = float('inf')
            if not match_indices:
                mcs = rdFMCS.FindMCS([mol, ref], threshold=1.0,
                                    ringMatchesRingOnly=False, completeRingsOnly=False,
                                    matchChiralTag=chirality)
                if not mcs:
                    return None
                patt = Chem.MolFromSmarts(mcs.smartsString)
                refMatch = ref.GetSubstructMatches(patt, uniquify=False)
                molMatch = mol.GetSubstructMatches(patt, uniquify=False)

                for ids_ref in refMatch:
                    ref_coord = get_coord(ref, ids_ref)
                    for ids_mol in molMatch:
                        mol_coord = get_coord(mol, ids_mol)
                        s = rmsd_calc(ref_coord, mol_coord)
                        if s < min_rmsd:
                            min_rmsd = s
                    del ref_coord  # Free memory
                del patt, refMatch, molMatch  # Free memory
            else:
                ref_coord = get_coord(ref)
                for ids in match_indices:
                    mol_coord = get_coord(mol, ids)
                    s = rmsd_calc(ref_coord, mol_coord)
                    if s < min_rmsd:
                        min_rmsd = s
                del ref_coord  # Free memory

            return round(min_rmsd, 3)
        except Exception as e:
            try:
                smiles1 = Chem.MolToSmiles(mol)
                smiles2 = Chem.MolToSmiles(ref)
                logging.error(f"Error calculating RMSD between {smiles1} and {smiles2}: {str(e)}")
            except:
                logging.error(f"Error calculating RMSD: {str(e)}")
            return None

    # Create a thread pool executor for this specific calculation
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            mol_name = getattr(mol, "_Name", "unknown")
            ref_name = getattr(ref, "_Name", "unknown")
            print(f"RMSD calculation timed out after {timeout}s for {mol_name} vs {ref_name}")
            return None

def calc_usr_similarity(mol1, mol2):
    """
    Calculate the USR-CAT similarity between two molecules.
    
    Parameters:
    -----------
    mol1 : rdkit.Chem.Mol
        The first molecule
    mol2 : rdkit.Chem.Mol
        The second molecule
        
    Returns:
    --------
    float
        The USR-CAT similarity value (between 0 and 1)
    """
    # Create Molecule objects for the two molecules
    shape_mol = oddt.toolkits.rdk.Molecule(mol1)
    shape_jmol = oddt.toolkits.rdk.Molecule(mol2)

    # Calculate the USR-CAT fingerprint for the two molecules
    mol_fp = oddt.shape.usr_cat(shape_mol)
    jmol_fp = oddt.shape.usr_cat(shape_jmol)

    # Calculate the USR-CAT similarity between the two fingerprints
    usr_sim = oddt.shape.usr_similarity(mol_fp, jmol_fp)
    
    return usr_sim

def calc_splif(molecule1: Chem.Mol, molecule2: Chem.Mol, pocket_file: Path) -> float:
    """
    Calculate the Protein-Ligand Interaction Fingerprint similarity between two molecules.

    Parameters:
    -----------
    molecule1 : rdkit.Chem.Mol
        The first molecule
    molecule2 : rdkit.Chem.Mol
        The second molecule
    pocket_file : Path
        Path to the pocket file (PDB format)

    Returns:
    --------
    float
        Rounded similarity score between the two molecules (between 0 and 1)
    """
    # Read the protein structure from the pocket file
    protein: None | oddt.toolkits.rdk.Molecule = next(oddt.toolkit.readfile("pdb", str(pocket_file)))
    protein.protein = True

    # Create Molecule objects for the two molecules
    splif_mol = oddt.toolkits.rdk.Molecule(molecule1)
    splif_jmol = oddt.toolkits.rdk.Molecule(molecule2)

    # Calculate the Simple Interaction Fingerprint (SIF) for the two molecules
    mol_fp = oddt.fingerprints.SimpleInteractionFingerprint(splif_mol, protein)
    jmol_fp = oddt.fingerprints.SimpleInteractionFingerprint(splif_jmol, protein)

    # Calculate the Tanimoto similarity between the two SIFs
    SPLIF_sim = oddt.fingerprints.dice(mol_fp, jmol_fp)

    # Round the similarity score to 3 decimal places
    return round(SPLIF_sim, 3)

class PairwiseMatrixComputer:
    def __init__(self, molecules, n_jobs=8, timeout=30, max_mols=None):
        """
        Initialize the PairwiseMatrixComputer.

        Parameters:
        -----------
        molecules : list
            List of molecules to compute pairwise values for (e.g., rdkit.Chem.Mol objects)
        n_jobs : int, optional
            Number of parallel jobs to use (default: 8)
        timeout : int, optional
            Maximum time in seconds for each pairwise calculation (default: 30)
        max_mols : int, optional
            Maximum number of molecules to process (default: None, processes all)
        """
        self.molecules = molecules[:max_mols] if max_mols else molecules
        self.n_jobs = n_jobs
        self.timeout = timeout

    def call_with_timeout(self, func, *args, **kwargs):
        """
        Execute a function with a timeout mechanism.

        Parameters:
        -----------
        func : callable
            The function to execute (e.g., pairwise_func)
        *args : tuple
            Positional arguments to pass to func (e.g., mol1, mol2)
        **kwargs : dict
            Keyword arguments to pass to func

        Returns:
        --------
        Any or None
            The result of func, or None if it times out or raises an exception
        """
        def worker():
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {str(e)}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                print(f"Timeout in {func.__name__} after {self.timeout}s")
                return None

    def compute_matrix(self, pairwise_func):
        """
        Compute a pairwise matrix using the provided pairwise function.

        Parameters:
        -----------
        pairwise_func : callable
            Function that takes two molecules and returns a value (e.g., float or None)

        Returns:
        --------
        numpy.ndarray
            A symmetric matrix of shape (n, n) containing pairwise values
        """
        start_time = time.time()
        n = len(self.molecules)
        print(f"Computing pairwise matrix for {n} molecules ({n*(n-1)//2} pairs)")

        def compute_pair(i, j):
            mol_i = self.molecules[i]
            mol_j = self.molecules[j]
            result = self.call_with_timeout(pairwise_func, mol_i, mol_j)

            pair_num = i * n + j - i * (i + 1) // 2
            if pair_num % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {pair_num} pairs in {elapsed:.1f}s ({pair_num/elapsed:.1f} pairs/s)")

            return (i, j, result)

        pairwise_matrix = np.full((n, n), np.nan)
        np.fill_diagonal(pairwise_matrix, 0)

        pairs = list(combinations(range(n), 2))
        results = Parallel(n_jobs=self.n_jobs)(delayed(compute_pair)(i, j) for i, j in pairs)

        for i, j, value in results:
            if value is not None:
                pairwise_matrix[i, j] = value
                pairwise_matrix[j, i] = value

        total_pairs = n * (n - 1) // 2
        completed_pairs = np.sum(~np.isnan(pairwise_matrix[np.triu_indices(n, k=1)]))
        print(f"Pairwise calculation complete! Computed {completed_pairs}/{total_pairs} pairs ({completed_pairs/total_pairs:.1%})")
        print(f"Total time: {time.time() - start_time:.1f}s")

        return pairwise_matrix

# def compute_rmsd_matrix(molecules, n_jobs=8, timeout=30, max_mols=None):
#     """
#     Compute RMSD matrix with timeouts for all molecule pairs.
#     
#     This function calculates a pairwise RMSD matrix for a set of molecules using
#     parallel processing. It includes progress tracking and timeout mechanisms for
#     each pairwise calculation.
#     
#     Parameters:
#     -----------
#     molecules : list of rdkit.Chem.Mol
#         List of molecules to compute pairwise RMSD for
#     n_jobs : int, optional
#         Number of parallel jobs to use (default: 8)
#     timeout : int, optional
#         Maximum time in seconds to wait for each pairwise calculation (default: 30)
#     max_mols : int, optional
#         Maximum number of molecules to process. If specified, only the first
#         max_mols molecules will be used (default: None)
#         
#     Returns:
#     --------
#     numpy.ndarray
#         A symmetric matrix of shape (n, n) where n is the number of molecules.
#         The matrix contains:
#         - RMSD values for each pair of molecules
#         - NaN values for failed calculations
#         - 0.0 on the diagonal (self-comparisons)
#         
#     Notes:
#     ------
#     - Progress information is printed every 100 pairs processed
#     - The function prints final statistics about completion rate and total time
#     - The matrix is symmetric (rmsd_matrix[i,j] = rmsd_matrix[j,i])
#     """

class OptimizedDBSCANClustering:
    """
    A class that handles DBSCAN clustering with automatic parameter optimization,
    with explicit control over noise levels and number of clusters.
    
    This class performs grid search for optimal DBSCAN parameters while ensuring
    the percentage of noise points stays below a user-defined threshold and
    the number of clusters doesn't exceed a maximum value.
    """
    
    def __init__(self, 
                 eps_range=(0.5, 5.0, 0.5),
                 min_samples_range=(2, 15),
                 max_noise_percent=15.0,
                 max_clusters=10,
                 use_dimensionality_reduction=True,
                 output_dimensions=[2, 3, 5, 10],
                 random_state=42,
                 n_jobs=-1,
                 score_weight=0.7,
                 verbose=True):
        """
        Initialize the clustering object with parameters for grid search.
        
        Parameters:
        -----------
        eps_range : tuple
            Range for epsilon parameter (start, stop, step)
        min_samples_range : tuple
            Range for min_samples parameter (min, max)
        max_noise_percent : float
            Maximum acceptable percentage of points classified as noise (default: 15%)
        max_clusters : int
            Maximum number of clusters allowed (default: 10)
        use_dimensionality_reduction : bool
            Whether to apply PCA dimensionality reduction before clustering (default: True)
        output_dimensions : list
            List of output dimensions to try for dimensionality reduction (default: [2, 3, 5, 10])
        random_state : int
            Random state for reproducibility (default: 42)
        n_jobs : int
            Number of parallel jobs (default: -1, all cores)
        score_weight : float
            Weight between silhouette score and noise percentage (0.7 means 70% 
            weight on silhouette score, 30% on minimizing noise)
        verbose : bool
            Whether to print progress information and results (default: True)
        """
        self.eps_range = eps_range
        self.min_samples_range = min_samples_range
        self.max_noise_percent = max_noise_percent
        self.max_clusters = max_clusters
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.output_dimensions = output_dimensions
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.score_weight = score_weight
        self.verbose = verbose
        
        # Initialize attributes to store results
        self.grid_search_results = None
        self.best_params = None
        self.best_labels = None
        self.embedding = None
        self.sil_per_cluster = None
        
    def fit(self, distance_matrix):
        """
        Perform the complete clustering process: grid search and final clustering.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
            
        Returns:
        --------
        numpy.ndarray
            Cluster labels for each data point
        """
        # Compute MDS embedding if dimensionality reduction is enabled
        if self.use_dimensionality_reduction:
            self._create_embedding(distance_matrix)
        
        # Perform grid search
        self._perform_grid_search(distance_matrix)
        
        # Calculate silhouette scores per cluster for the best solution
        if self.best_labels is not None and np.any(self.best_labels != -1):
            self._calculate_silhouette_per_cluster(distance_matrix)
        
        # Print report if verbose
        if self.verbose:
            self._print_report()
        
        # Return the best labels
        return self.best_labels
    
    def _create_embedding(self, distance_matrix):
        """
        Create embedding from distance matrix for dimensionality reduction.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        """
        if self.verbose:
            print("Computing MDS embedding from distance matrix...")
        
        # Convert distance matrix to similarity matrix for better MDS results
        similarity_matrix = 1 / (1 + distance_matrix)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Create MDS embedding
        mds = MDS(
            n_components=min(distance_matrix.shape[0]-1, max(self.output_dimensions)),
            dissimilarity='precomputed',
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        mds_embedding = mds.fit_transform(distance_matrix)
        
        # Scale the embedding
        scaler = StandardScaler()
        self.embedding = scaler.fit_transform(mds_embedding)
    
    def _perform_grid_search(self, distance_matrix):
        """
        Perform grid search for optimal DBSCAN parameters with noise control.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        """
        if self.verbose:
            print("Starting grid search for optimal DBSCAN parameters...")
            print(f"Maximum allowed noise: {self.max_noise_percent:.1f}%")
            print(f"Maximum allowed clusters: {self.max_clusters}")
        
        # Initialize variables to track best results
        best_combined_score = -np.inf
        self.best_params = None
        self.best_labels = None
        best_silhouette = -1
        best_noise_percent = 100
        best_n_clusters = 0
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(distance_matrix.shape[0])
        
        # Set up progress tracking
        if self.verbose:
            print(f"Total parameter combinations to try: {len(param_combinations)}")
            iterator = tqdm(param_combinations)
        else:
            iterator = param_combinations
        
        # Track all results
        all_results = []
        
        # Perform grid search
        for params in iterator:
            # Apply clustering with current parameters
            labels, silhouette, n_clusters, n_noise, noise_percent, X_reduced = self._apply_clustering(
                distance_matrix, params
            )
            
            # Skip invalid results
            if n_clusters <= 1 or silhouette < 0:
                result = {
                    'params': params,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percent': noise_percent,
                    'silhouette': -1,
                    'combined_score': -np.inf
                }
                all_results.append(result)
                continue
                
            # Skip if too many clusters
            if n_clusters > self.max_clusters:
                result = {
                    'params': params,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percent': noise_percent,
                    'silhouette': silhouette,
                    'combined_score': -np.inf,  # Mark as invalid due to too many clusters
                    'reason': 'too_many_clusters'
                }
                all_results.append(result)
                continue
            
            # Calculate combined score that balances silhouette and noise
            # Only consider parameter sets that meet the noise threshold
            if noise_percent <= self.max_noise_percent:
                # Normalize silhouette to [0, 1] (it's already in [-1, 1])
                norm_silhouette = (silhouette + 1) / 2
                # Normalize noise_percent to [0, 1] where 0 is best (no noise)
                # and 1 is worst (100% noise)
                norm_noise = noise_percent / 100
                
                # Combined score: weighted average of silhouette (higher is better) 
                # and negated noise (lower is better)
                combined_score = (self.score_weight * norm_silhouette - 
                                 (1 - self.score_weight) * norm_noise)
                
                result = {
                    'params': params,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percent': noise_percent,
                    'silhouette': silhouette,
                    'combined_score': combined_score,
                    'labels': labels,
                    'reduced_data': X_reduced
                }
                all_results.append(result)
                
                # Update best score if better
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    self.best_params = params.copy()
                    self.best_labels = labels
                    best_silhouette = silhouette
                    best_noise_percent = noise_percent
                    best_n_clusters = n_clusters
            else:
                # Store the result but don't consider it for best
                result = {
                    'params': params,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percent': noise_percent,
                    'silhouette': silhouette,
                    'combined_score': -np.inf,  # Mark as invalid due to too much noise
                    'reason': 'too_much_noise'
                }
                all_results.append(result)
        
        # Filter valid results
        valid_results = [r for r in all_results if r.get('combined_score', -np.inf) > -np.inf]
        
        # Sort by combined score
        top_results = sorted(valid_results, key=lambda x: x.get('combined_score', -np.inf), reverse=True)[:10]
        
        # Store results
        self.grid_search_results = {
            'best_params': self.best_params,
            'best_silhouette': best_silhouette,
            'best_noise_percent': best_noise_percent,
            'best_n_clusters': best_n_clusters,
            'best_labels': self.best_labels,
            'top_results': top_results,
            'all_results': all_results
        }
        
        # Visualization of best result
        if self.verbose and self.best_params and 'labels' in next(
            (r for r in all_results if r['params'] == self.best_params), {}
        ):
            self._visualize_best_result(top_results[0] if top_results else None)
    
    def _generate_parameter_combinations(self, n_samples):
        """
        Generate all parameter combinations for grid search.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples in the dataset
        
        Returns:
        --------
        list
            List of parameter dictionaries
        """
        param_combinations = []
        
        # Generate eps values
        eps_values = np.arange(self.eps_range[0], self.eps_range[1], self.eps_range[2])
        
        # Generate min_samples values
        min_samples_values = np.arange(self.min_samples_range[0], self.min_samples_range[1])
        
        # Direct DBSCAN on distance matrix
        for eps in eps_values:
            for min_samples in min_samples_values:
                param_combinations.append({
                    'reduction': 'none',
                    'dimensions': 'original',
                    'eps': eps,
                    'min_samples': min_samples
                })
        
        # Add dimensionality reduction methods if enabled
        if self.use_dimensionality_reduction:
            for dim in self.output_dimensions:
                if dim >= n_samples:
                    continue  # Skip if dimension is too large
                
                for eps in eps_values:
                    for min_samples in min_samples_values:
                        param_combinations.append({
                            'reduction': 'pca',
                            'dimensions': dim,
                            'eps': eps,
                            'min_samples': min_samples
                        })
        
        return param_combinations
    
    def _apply_clustering(self, distance_matrix, params):
        """
        Apply clustering with given parameters.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        params : dict
            Dictionary of parameters
        
        Returns:
        --------
        tuple
            (labels, silhouette_score, n_clusters, n_noise, noise_percent, reduced_data)
        """
        reduction = params['reduction']
        dimensions = params['dimensions']
        eps = params['eps']
        min_samples = params['min_samples']
        
        # Apply dimensionality reduction
        if reduction == 'none':
            # Use original distance matrix
            clustering = DBSCAN(
                eps=eps, 
                min_samples=min_samples, 
                metric='precomputed'
            ).fit(distance_matrix)
            X_reduced = None
        else:
            # Apply dimensionality reduction
            if reduction == 'pca':
                # Use PCA on embedding
                reducer = PCA(n_components=dimensions, random_state=self.random_state)
                X_reduced = reducer.fit_transform(self.embedding)
            
            # Run DBSCAN on reduced data
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_reduced)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_percent = 100 * n_noise / len(labels)
        
        # Skip if all points are noise or only one cluster
        if n_clusters <= 1:
            return (labels, -1, n_clusters, n_noise, noise_percent, X_reduced)
        
        # Calculate silhouette score
        silhouette = -1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if reduction == 'none':
                    # Use distance matrix for silhouette
                    non_noise_idx = [i for i, l in enumerate(labels) if l != -1]
                    if len(non_noise_idx) > 1:
                        filtered_distance = distance_matrix[np.ix_(non_noise_idx, non_noise_idx)]
                        filtered_labels = labels[non_noise_idx]
                        silhouette = silhouette_score(
                            filtered_distance, 
                            filtered_labels, 
                            metric='precomputed'
                        )
                else:
                    # Use reduced data for silhouette
                    mask = labels != -1
                    if np.sum(mask) > 1:  # Need at least 2 points
                        silhouette = silhouette_score(X_reduced[mask], labels[mask])
        except Exception as e:
            if self.verbose:
                print(f"Silhouette calculation failed for {params}: {str(e)}")
        
        return (labels, silhouette, n_clusters, n_noise, noise_percent, X_reduced)
    
    def _calculate_silhouette_per_cluster(self, distance_matrix):
        """
        Calculate silhouette score for each cluster in the best clustering.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        """
        labels = self.best_labels
        best_params = self.best_params
        
        # Skip if no valid clustering found
        if labels is None or best_params is None:
            self.sil_per_cluster = {}
            return
        
        try:
            # Get non-noise points
            non_noise_mask = labels != -1
            non_noise_indices = np.where(non_noise_mask)[0]
            
            if len(non_noise_indices) <= 1:
                self.sil_per_cluster = {}
                return
                
            # Calculate silhouette values per sample
            if best_params['reduction'] == 'none':
                # Use distance matrix for silhouette calculation
                filtered_distance = distance_matrix[np.ix_(non_noise_indices, non_noise_indices)]
                filtered_labels = labels[non_noise_indices]
                
                # Calculate silhouette values for each sample
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample_silhouette_values = silhouette_samples(
                        filtered_distance, 
                        filtered_labels, 
                        metric='precomputed'
                    )
            else:
                # For dimensionality reduction methods, we need the reduced data
                reduction = best_params['reduction']
                dimensions = best_params['dimensions']
                
                # Get or recompute the reduced data
                if reduction == 'pca':
                    reducer = PCA(n_components=dimensions, random_state=self.random_state)
                    X_reduced = reducer.fit_transform(self.embedding)
                
                # Filter out noise points
                X_filtered = X_reduced[non_noise_mask]
                filtered_labels = labels[non_noise_mask]
                
                # Calculate silhouette values for each sample
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample_silhouette_values = silhouette_samples(
                        X_filtered, 
                        filtered_labels
                    )
            
            # Calculate average silhouette score per cluster
            self.sil_per_cluster = {}
            for cluster_label in np.unique(filtered_labels):
                cluster_silhouette_values = sample_silhouette_values[filtered_labels == cluster_label]
                self.sil_per_cluster[int(cluster_label)] = float(np.mean(cluster_silhouette_values))
            
            # Add to grid search results
            self.grid_search_results['sil_per_cluster'] = self.sil_per_cluster
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating silhouette per cluster: {str(e)}")
            self.sil_per_cluster = {}
    
    def _print_report(self):
        """Print a report of the grid search results."""
        if self.grid_search_results is None or self.best_params is None:
            print("No valid clustering results found!")
            return
            
        print("\n===== DBSCAN Clustering Report =====")
        print(f"Best silhouette score: {self.grid_search_results['best_silhouette']:.4f}")
        print(f"Noise percentage: {self.grid_search_results['best_noise_percent']:.1f}%")
        print(f"Number of clusters: {self.grid_search_results['best_n_clusters']}")
        
        best = self.best_params
        print(f"Best parameters:")
        print(f"  - Dimensionality reduction: {best['reduction']}")
        print(f"  - Dimensions: {best['dimensions']}")
        print(f"  - Epsilon (eps): {best['eps']}")
        print(f"  - Min samples: {best['min_samples']}")
        
        if self.best_labels is not None:
            n_clusters = len(set(self.best_labels)) - (1 if -1 in self.best_labels else 0)
            n_noise = list(self.best_labels).count(-1)
            noise_percent = 100 * n_noise / len(self.best_labels)
            
            print(f"Clustering results:")
            print(f"  - Number of clusters: {n_clusters}")
            print(f"  - Noise points: {n_noise} ({noise_percent:.1f}%)")
            
            # Print silhouette score per cluster
            if self.sil_per_cluster:
                print("\nSilhouette score per cluster:")
                for cluster, score in sorted(self.sil_per_cluster.items()):
                    print(f"  - Cluster {cluster}: {score:.4f}")
        
        # Print top 5 results
        top_results = self.grid_search_results.get('top_results', [])
        if top_results:
            print("\nTop 5 parameter combinations:")
            for i, result in enumerate(top_results[:5]):
                p = result['params']
                print(f"{i+1}. Score: {result['silhouette']:.3f}, Noise: {result['noise_percent']:.1f}%, "
                      f"Clusters: {result['n_clusters']}, "
                      f"Method: {p['reduction']}, Dims: {p['dimensions']}, "
                      f"Eps: {p['eps']}, MinSamples: {p['min_samples']}")
        
        print("====================================")
    
    def _visualize_best_result(self, best_result):
        """
        Visualize the best clustering result.
        
        Parameters:
        -----------
        best_result : dict
            Dictionary containing the best result details
        """
        if best_result is None:
            return
            
        # Get the best clustering result
        best_params = best_result['params']
        best_labels = best_result['labels']
        
        # Create a visualization of the best clustering
        if best_params['reduction'] != 'none':
            if best_result.get('reduced_data') is not None and best_result['reduced_data'].shape[1] > 1:
                # Use the reduced data directly if available
                X_viz = best_result['reduced_data']
                if X_viz.shape[1] > 2:
                    # Apply PCA to get 2D visualization
                    viz_reducer = PCA(n_components=2, random_state=self.random_state)
                    X_viz = viz_reducer.fit_transform(X_viz)
            else:
                # Apply PCA with 2D for visualization
                viz_reducer = PCA(n_components=2, random_state=self.random_state)
                X_viz = viz_reducer.fit_transform(self.embedding)
            
            # Plot the clusters
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1], c=best_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f"Best Clustering: {best_params['reduction'].upper()} + DBSCAN\n"
                     f"Dimensions: {best_params['dimensions']}, Eps: {best_params['eps']}, "
                     f"MinSamples: {best_params['min_samples']}\n"
                     f"Silhouette: {best_result['silhouette']:.3f}, "
                     f"Noise: {best_result['noise_percent']:.1f}%, "
                     f"Clusters: {best_result['n_clusters']}")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.show()
    
    def get_results(self):
        """
        Get the complete results of the clustering process.
        
        Returns:
        --------
        dict
            Dictionary containing best parameters, labels, silhouette scores per cluster,
            and full grid search results
        """
        return {
            'best_params': self.best_params,
            'best_labels': self.best_labels,
            'sil_per_cluster': self.sil_per_cluster,
            'grid_search_results': self.grid_search_results
        }

class OptimizedKMedoidsClustering:
    """
    A class that performs K-Medoids clustering with automatic parameter optimization,
    with explicit control over the number of clusters.
    
    This class conducts a grid search over the number of clusters (K) and optional
    dimensionality reduction methods, aiming for the best clustering quality based
    on silhouette score. It is designed to be highly automatic, requiring only a
    distance matrix as input with default settings.
    """
    
    def __init__(self, 
                 k_range=(2, 10),
                 use_dimensionality_reduction=True,
                 output_dimensions=[2, 3, 5, 10],
                 random_state=42,
                 n_jobs=-1,
                 verbose=True):
        """
        Initialize the clustering object with parameters for grid search.
        
        Parameters:
        -----------
        k_range : tuple
            Range of K values (number of clusters) to try (min, max)
        use_dimensionality_reduction : bool
            Whether to apply PCA dimensionality reduction before clustering (default: True)
        output_dimensions : list
            List of output dimensions to try for dimensionality reduction (default: [2, 3, 5, 10])
        random_state : int
            Random state for reproducibility (default: 42)
        n_jobs : int
            Number of parallel jobs (default: -1, all cores)
        verbose : bool
            Whether to print progress information and results (default: True)
        """
        self.k_range = k_range
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.output_dimensions = output_dimensions
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Attributes to store results (consistent with OptimizedDBSCANClustering)
        self.grid_search_results = None
        self.best_params = None
        self.best_labels = None
        self.embedding = None
        self.sil_per_cluster = None
    
    def fit(self, distance_matrix):
        """
        Perform the complete clustering process: grid search and final clustering.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
            
        Returns:
        --------
        numpy.ndarray
            Cluster labels for each data point
        """
        # Compute MDS embedding if dimensionality reduction is enabled
        if self.use_dimensionality_reduction:
            self._create_embedding(distance_matrix)
        
        # Perform grid search to find the best parameters
        self._perform_grid_search(distance_matrix)
        
        # Calculate silhouette scores per cluster for the best solution
        if self.best_labels is not None:
            self._calculate_silhouette_per_cluster(distance_matrix)
        
        # Print report if verbose
        if self.verbose:
            self._print_report()
        
        return self.best_labels
    
    def _create_embedding(self, distance_matrix):
        """
        Create embedding from distance matrix for dimensionality reduction.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        """
        if self.verbose:
            print("Computing MDS embedding from distance matrix...")
        
        # Convert distance matrix to similarity matrix for better MDS results
        similarity_matrix = 1 / (1 + distance_matrix)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        mds = MDS(
            n_components=min(distance_matrix.shape[0] - 1, max(self.output_dimensions)),
            dissimilarity='precomputed',
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        mds_embedding = mds.fit_transform(distance_matrix)
        scaler = StandardScaler()
        self.embedding = scaler.fit_transform(mds_embedding)
    
    def _perform_grid_search(self, distance_matrix):
        """
        Perform grid search for optimal K-Medoids parameters.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        """
        param_combinations = self._generate_parameter_combinations(distance_matrix.shape[0])
        
        if self.verbose:
            print(f"Starting grid search for optimal K-Medoids parameters...")
            print(f"Total parameter combinations to try: {len(param_combinations)}")
            iterator = tqdm(param_combinations)
        else:
            iterator = param_combinations
        
        best_silhouette = -1
        self.best_params = None
        self.best_labels = None
        all_results = []
        
        for params in iterator:
            labels, silhouette, n_clusters, n_noise, noise_percent, X_reduced = self._apply_clustering(distance_matrix, params)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                self.best_params = params.copy()
                self.best_labels = labels
            all_results.append({
                'params': params,
                'silhouette': silhouette,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_percent': noise_percent,
                'labels': labels,
                'reduced_data': X_reduced
            })
        
        # Sort results by silhouette score
        top_results = sorted(all_results, key=lambda x: x['silhouette'], reverse=True)[:10]
        
        self.grid_search_results = {
            'best_params': self.best_params,
            'best_silhouette': best_silhouette,
            'best_labels': self.best_labels,
            'top_results': top_results,
            'all_results': all_results
        }
        
        # Visualization of best result
        if self.verbose and self.best_params:
            self._visualize_best_result(self.grid_search_results['top_results'][0])
    
    def _generate_parameter_combinations(self, n_samples):
        """
        Generate all parameter combinations for grid search.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples in the dataset
        
        Returns:
        --------
        list
            List of parameter dictionaries
        """
        param_combinations = []
        
        # K-Medoids directly on distance matrix
        for k in np.arange(self.k_range[0], self.k_range[1] + 1):
            if k > n_samples:
                continue
            param_combinations.append({
                'reduction': 'none',
                'dimensions': 'original',
                'k': k
            })
        
        # Add dimensionality reduction methods if enabled
        if self.use_dimensionality_reduction:
            for dim in self.output_dimensions:
                if dim >= n_samples:
                    continue
                for k in np.arange(self.k_range[0], self.k_range[1] + 1):
                    if k > n_samples:
                        continue
                    param_combinations.append({
                        'reduction': 'pca',
                        'dimensions': dim,
                        'k': k
                    })
        return param_combinations
    
    def _apply_clustering(self, distance_matrix, params):
        """
        Apply clustering with given parameters.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        params : dict
            Dictionary of parameters
        
        Returns:
        --------
        tuple
            (labels, silhouette_score, n_clusters, n_noise, noise_percent, reduced_data)
        """
        reduction = params['reduction']
        dimensions = params['dimensions']
        k = params['k']
        
        if reduction == 'none':
            clusterer = KMedoids(
                n_clusters=k,
                metric='precomputed',
                random_state=self.random_state
            )
            labels = clusterer.fit_predict(distance_matrix)
            X_reduced = None
            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed') if len(set(labels)) > 1 else -1
            n_clusters = k
            n_noise = 0  # K-Medoids doesn't have noise points
            noise_percent = 0.0
        else:
            if reduction == 'pca':
                reducer = PCA(n_components=dimensions, random_state=self.random_state)
                X_reduced = reducer.fit_transform(self.embedding)
            clusterer = KMedoids(n_clusters=k, random_state=self.random_state)
            labels = clusterer.fit_predict(X_reduced)
            silhouette = silhouette_score(X_reduced, labels) if len(set(labels)) > 1 else -1
            n_clusters = k
            n_noise = 0  # K-Medoids doesn't have noise points
            noise_percent = 0.0
        
        return labels, silhouette, n_clusters, n_noise, noise_percent, X_reduced
    
    def _calculate_silhouette_per_cluster(self, distance_matrix):
        """
        Calculate silhouette score for each cluster in the best clustering.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix for clustering
        """
        if self.best_labels is None or self.best_params is None:
            self.sil_per_cluster = {}
            return
        
        try:
            if self.best_params['reduction'] == 'none':
                sample_silhouette_values = silhouette_samples(distance_matrix, self.best_labels, metric='precomputed')
            else:
                reduction = self.best_params['reduction']
                dimensions = self.best_params['dimensions']
                if reduction == 'pca':
                    reducer = PCA(n_components=dimensions, random_state=self.random_state)
                    X_reduced = reducer.fit_transform(self.embedding)
                sample_silhouette_values = silhouette_samples(X_reduced, self.best_labels)
            
            self.sil_per_cluster = {}
            for cluster_label in np.unique(self.best_labels):
                cluster_sil_vals = sample_silhouette_values[self.best_labels == cluster_label]
                self.sil_per_cluster[int(cluster_label)] = float(np.mean(cluster_sil_vals))
        except Exception as e:
            if self.verbose:
                print(f"Error calculating silhouette per cluster: {str(e)}")
            self.sil_per_cluster = {}
    
    def _print_report(self):
        """Print a report of the grid search results."""
        if self.grid_search_results is None or self.best_params is None:
            print("No valid clustering results found!")
            return
        
        print("\n===== K-Medoids Clustering Report =====")
        print(f"Best silhouette score: {self.grid_search_results['best_silhouette']:.4f}")
        print(f"Number of clusters: {self.best_params['k']}")
        
        best = self.best_params
        print(f"Best parameters:")
        print(f"  - Dimensionality reduction: {best['reduction']}")
        print(f"  - Dimensions: {best['dimensions']}")
        print(f"  - K: {best['k']}")
        
        if self.sil_per_cluster:
            print("\nSilhouette score per cluster:")
            for cluster, score in sorted(self.sil_per_cluster.items()):
                print(f"  - Cluster {cluster}: {score:.4f}")
        
        top_results = self.grid_search_results.get('top_results', [])
        if top_results:
            print("\nTop 5 parameter combinations:")
            for i, result in enumerate(top_results[:5]):
                p = result['params']
                print(f"{i+1}. Score: {result['silhouette']:.3f}, "
                      f"Method: {p['reduction']}, Dims: {p['dimensions']}, K: {p['k']}")
        
        print("====================================")
    
    def _visualize_best_result(self, best_result):
        """
        Visualize the best clustering result.
        
        Parameters:
        -----------
        best_result : dict
            Dictionary containing the best result details
        """
        if best_result is None:
            return
        
        # Get the best clustering result
        best_params = best_result['params']
        best_labels = best_result['labels']
        
        # Create a visualization of the best clustering
        if best_params['reduction'] != 'none':
            if best_result.get('reduced_data') is not None and best_result['reduced_data'].shape[1] > 1:
                # Use the reduced data directly if available
                X_viz = best_result['reduced_data']
                if X_viz.shape[1] > 2:
                    # Apply PCA to get 2D visualization
                    viz_reducer = PCA(n_components=2, random_state=self.random_state)
                    X_viz = viz_reducer.fit_transform(X_viz)
            else:
                # Apply PCA with 2D for visualization
                viz_reducer = PCA(n_components=2, random_state=self.random_state)
                X_viz = viz_reducer.fit_transform(self.embedding)
            
            # Plot the clusters
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1], c=best_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f"Best Clustering: {best_params['reduction'].upper()} + K-Medoids\n"
                     f"Dimensions: {best_params['dimensions']}, K: {best_params['k']}\n"
                     f"Silhouette: {best_result['silhouette']:.3f}")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.show()
    
    def get_results(self):
        """
        Get the complete results of the clustering process.
        
        Returns:
        --------
        dict
            Dictionary containing best parameters, labels, silhouette scores per cluster,
            and full grid search results
        """
        return {
            'best_params': self.best_params,
            'best_labels': self.best_labels,
            'sil_per_cluster': self.sil_per_cluster,
            'grid_search_results': self.grid_search_results
        }