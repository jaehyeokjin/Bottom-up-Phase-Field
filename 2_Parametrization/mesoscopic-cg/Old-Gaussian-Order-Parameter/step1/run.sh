source /etc/profile
module load intel
module load lapack


module load gsl
module load mkl
./FitGLE.x ../trj/aa.trj 2048
