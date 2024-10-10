#!/bin/sh

#SBATCH --qos=gavoth
#SBATCH --partition=gavoth-brw
#SBATCH --job-name=water
#SBATCH --time=20:00:00
#SBATCH --verbose

export OMP_NUM_THREADS=1
source /etc/profile
module load gsl/1.16
module load mkl
module load intel/12.1

./newfm.x -l ../../../water_interface.lammpstrj
