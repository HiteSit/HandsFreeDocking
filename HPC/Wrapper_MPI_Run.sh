#!/bin/bash -l

#SBATCH --job-name=VS_Gnina
#SBATCH --output=VS_Gnina.log
#SBATCH --error=VS_Gnina.error
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --hint=multithread
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

#SBATCH --time=00:30:00
#SBATCH --account=project_465001750

# Load the required environment
export PATH="/scratch/project_465001750/Singularity_Envs/cheminf_mpi/env_cheminf_mpi/bin:$PATH"

# Make sure gnina is executable
if [ -f ./gnina ]; then
    chmod +x ./gnina
fi

# Run the MPI docking script
srun python Wrapper_MPI.py