export OMP_NUM_THREADS=1
source /etc/profile
module load intel
module load intelmpi
module load fftw3

mpirun -np 200 ~/lammps/src/lmp_midway -in in.mscg
