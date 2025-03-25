#!/bin/bash
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --nodes=4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=500mb
#SBATCH -t 00:05:00
#SBATCH -o 4T-1R-4N
#SBATCH -e errfile
export OMP_NUM_THREADS=4
srun --mpi=$HPC_PMIX ./test input.txt