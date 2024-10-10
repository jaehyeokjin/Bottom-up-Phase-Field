#!/bin/sh

#SBATCH --partition=sandyb
#SBATCH --job-name=water
#SBATCH --time=15:00:00
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export OMP_NUM_THREADS=1
source /etc/profile
module load gsl/1.16
module load mkl
module load intel/12.1

./rangefinder.x -l ../../cg.lammpstrj
