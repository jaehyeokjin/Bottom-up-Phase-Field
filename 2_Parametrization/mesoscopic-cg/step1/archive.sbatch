#!/bin/sh

#SBATCH --account=pi-gavoth
#SBATCH --partition=broadwl-lc
#SBATCH --job-name=water
#SBATCH --time=6:00:00
#SBATCH --verbose
#SBATCH --ntasks-per-node=1
export OMP_NUM_THREADS=1
source /etc/profile
module load intel
module load lapack


module load gsl
module load mkl
./FitGLE.x ../trj/aa.trj 2048
