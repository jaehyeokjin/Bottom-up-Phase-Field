#include "FitGLE.h"
#include "comm.h"
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cassert>
#include <gsl/gsl_bspline.h>
#include <lapacke.h>
#include <iostream>
#include <cstdlib>

using namespace FITGLE_NS;

FitGLE::FitGLE(int argc, char** argv)
{
    if (argc != 3)
    {
        printf("./FitGLE.x [trajectory Filename] [number of Particles]\n");
    }

    assert(argc == 3);
    printf("Initializing FitGLE parameters...\n");

    // parsing the configuration parameters
    info = std::make_shared<InputParameters>();
    VAR_BEGIN
      GET_REAL(info->start)
      GET_REAL(info->end)
      GET_REAL(info->boxLengthX)
      GET_REAL(info->boxLengthY)
      GET_REAL(info->boxLengthZ)
      GET_REAL(info->outputPrecision)
      GET_INT(info->splineOrder)
      GET_INT(info->numSplines)
      GET_INT(info->steps)
    VAR_END
           
    printf("set up trajectory files\n");
    trajFrame = std::make_shared<Frame>(atoi(argv[2]), argv[1]);
    // Initialize the Normal Equation matrix and vectors
    // Set up the size of splines according to order and numbers
    printf("set up b-spline data structures\n");
    int numBreaks = info->numSplines + 2 - info->splineOrder;
    normalVector.resize(info->numSplines);
    splineCoefficients.resize(info->numSplines);
    normalMatrix.resize(info->numSplines);
    // Per particle vectors (new in this work)
    density.resize(trajFrame->numParticles);
    density_square.resize(trajFrame->numParticles);
    density_sq_factor.resize(trajFrame->numParticles);
    printf("set up containers\n");
    for (auto&& i : normalMatrix)
    {
        i.resize(info->numSplines);
    }

    // Initialize the spline set up
    bw = gsl_bspline_alloc(info->splineOrder, numBreaks);
    splineValue = gsl_vector_alloc(info->numSplines);
    gsl_bspline_knots_uniform(info->start, info->end, bw);

    printf("finishing configuration, entering normal equation accumulation\n");
}

FitGLE::~FitGLE()
{
    gsl_bspline_free(bw);
    gsl_vector_free(splineValue);
    printf("Exiting the Fitting GLE process...\n");
}

inline double FitGLE::distance(std::vector<double> & A, std::vector<double> & B)
{
    double dx = A[0] - B[0];
    if (dx > 0.5 * info->boxLengthX) dx = dx - info->boxLengthX;
    if (dx < -0.5 * info->boxLengthX) dx = dx + info->boxLengthX;
    double dy = A[1] - B[1];
    if (dy > 0.5 * info->boxLengthY) dy = dy - info->boxLengthY;
    if (dy < -0.5 * info->boxLengthY) dy = dy + info->boxLengthY;
    double dz = A[2] - B[2];
    if (dz > 0.5 * info->boxLengthZ) dz = dz - info->boxLengthZ;
    if (dz < -0.5 * info->boxLengthZ) dz = dz + info->boxLengthZ;
  
    return sqrt(dx*dx + dy*dy + dz*dz);
}

inline std::vector<double> FitGLE::parallelVelocity(int i, int j)
{
    double dx = trajFrame->positions[i][0] - trajFrame->positions[j][0];
    if (dx > 0.5 * info->boxLengthX) dx = dx - info->boxLengthX;
    if (dx < -0.5 * info->boxLengthX) dx = dx + info->boxLengthX;
    double dy = trajFrame->positions[i][1] - trajFrame->positions[j][1];
    if (dy > 0.5 * info->boxLengthY) dy = dy - info->boxLengthY;
    if (dy < -0.5 * info->boxLengthY) dy = dy + info->boxLengthY;
    double dz = trajFrame->positions[i][2] - trajFrame->positions[j][2];
    if (dz > 0.5 * info->boxLengthZ) dz = dz - info->boxLengthZ;
    if (dz < -0.5 * info->boxLengthZ) dz = dz + info->boxLengthZ;

    std::vector<double> vij(3, 0.0);
    vij[0] = dx;
    vij[1] = dy;
    vij[2] = dz;
     
    return vij;
}

inline std::vector<double> FitGLE::parallelUnitVector(int i, int j)
{
    double dx = trajFrame->positions[i][0] - trajFrame->positions[j][0];
    if (dx > 0.5 * info->boxLengthX) dx = dx - info->boxLengthX;
    if (dx < -0.5 * info->boxLengthX) dx = dx + info->boxLengthX;
    double dy = trajFrame->positions[i][1] - trajFrame->positions[j][1];
    if (dy > 0.5 * info->boxLengthY) dy = dy - info->boxLengthY;
    if (dy < -0.5 * info->boxLengthY) dy = dy + info->boxLengthY;
    double dz = trajFrame->positions[i][2] - trajFrame->positions[j][2];
    if (dz > 0.5 * info->boxLengthZ) dz = dz - info->boxLengthZ;
    if (dz < -0.5 * info->boxLengthZ) dz = dz + info->boxLengthZ;

    double rij = sqrt(dx*dx + dy*dy + dz*dz);
    std::vector<double> eij;
    eij.push_back(dx / rij);
    eij.push_back(dy / rij);
    eij.push_back(dz / rij);
    //printf("%lf %lf %lf d %lf %lf %lf\n", dx, dy, dz, trajFrame->positions[i][0], trajFrame->positions[j][0], info->boxLength);
    return eij;
}

// Accumulate the normal equation for this particular frame
void FitGLE::accumulateNormalEquation()
{
    //printf("Starting the normal equation construction\n");
    int nall = trajFrame->numParticles;
    int nSplines = info->numSplines;
    std::vector<std::vector<double> > frameMatrix(nSplines, std::vector<double>(3*nall));
    std::vector<std::vector<double> > frameVector(nall, std::vector<double>(3));
    std::vector<std::vector<double> > density_grad(trajFrame->numParticles, std::vector <double>(3, 0.0));
    double normalFactor = 1.0 / info->steps;
    // Step 1: Accumulate density
    for (int i=0; i<nall; i++)
    {
        double density_value = 0.0;
	double density_sq_factor_value = 0.0;
        double density_sq_x = 0.0;
        double density_sq_y = 0.0;
        double density_sq_z = 0.0;
        for (int j=0;j<nall;j++)
        {
            if (i!=j)
            {
                double pair_distance = distance(trajFrame->positions[i],trajFrame->positions[j]);
                density_value += exp(-1.0*pair_distance*pair_distance/800.0);
                density_sq_x += -1.0*exp(-1.0*pair_distance*pair_distance/800.0)*2.0/800.0*parallelVelocity(i,j)[0];
                density_sq_y += -1.0*exp(-1.0*pair_distance*pair_distance/800.0)*2.0/800.0*parallelVelocity(i,j)[1];
                density_sq_z += -1.0*exp(-1.0*pair_distance*pair_distance/800.0)*2.0/800.0*parallelVelocity(i,j)[2];
		density_sq_factor_value += exp(-1.0*pair_distance*pair_distance/800.0)*((2.0/800.0)*(2.0/800.0)* pair_distance*pair_distance - 2.0/800.0);
            }
        }
	density_grad[i][0] = density_sq_x;
        density_grad[i][1] = density_sq_y;
        density_grad[i][2] = density_sq_z;
        double density_sq = density_sq_x*density_sq_x + density_sq_y*density_sq_y + density_sq_z*density_sq_z;
        density[i] = density_value;
        density_square[i] = density_sq * (0.25*0.25*(1.0-tanh((density_value-1132.0)/2.0)*tanh((density_value-1132.0)/2.0))*(1.0-tanh((density_value-1132.0)/2.0)*tanh((density_value-1132.0)/2.0)));
	density_sq_factor[i] = density_sq_factor_value;
    }

    // Step 2: Setting up Relevant variables and construct force matching matrix
    for (int i=0; i<nall; i++)
    {
        double phi = 0.5*(1.0-tanh((density[i]-1132.0)/2.0));
	double p_value = -0.7137860;
        double q_value = 9.544258;
        double epsilon_value = 1.004265;
        double grad_x = 0.0;
        double grad_y = 0.0;
        double grad_z = 0.0;
        double prefactor_i = -0.25*(1.0-tanh((density[i]-1132.0)/2.0)*tanh((density[i]-1132.0)/2.0));
	double deriv_prefactor_i = 0.25*(1.0-tanh((density[i]-1132.0)/2.0)*tanh((density[i]-1132.0)/2.0))*tanh((density[i]-1132.0)/2.0);
        for (int j = 0; j<nall; j++)
        {
            if (i!=j)
            {
                double prefactor_j = -0.25*(1.0-tanh((density[j]-1132.0)/2.0)*tanh((density[j]-1132.0)/2.0));
                double phi_j = 0.5*(1.0-tanh((density[j]-1132.0)/2.0));
                double pair_distance = distance(trajFrame->positions[i],trajFrame->positions[j]);
                double increment_x = exp(-1.0*pair_distance*pair_distance/800.0)*2.0/800.0*parallelVelocity(i,j)[0];
                double increment_y = exp(-1.0*pair_distance*pair_distance/800.0)*2.0/800.0*parallelVelocity(i,j)[1];
                double increment_z = exp(-1.0*pair_distance*pair_distance/800.0)*2.0/800.0*parallelVelocity(i,j)[2];
                grad_x += increment_x;
                grad_y += increment_y;
                grad_z += increment_z;
		double deriv_prefactor_j = 0.25*(1.0-tanh((density[j]-1132.0)/2.0)*tanh((density[j]-1132.0)/2.0))*tanh((density[j]-1132.0)/2.0);
                double nabla_inside_1 = -1.0*deriv_prefactor_j * (increment_x*density_grad[j][0]+increment_y*density_grad[j][1]+increment_z*density_grad[j][2]);
                double nabla_inside_2 = prefactor_j * exp(-1.0*pair_distance*pair_distance/800.0) * ((2.0/800.0)*(2.0/800.0)*pair_distance*pair_distance +2.0/800.0);
                double nabla_ij_x = -1.0 * prefactor_j * density_grad[j][0] * (nabla_inside_1+nabla_inside_2);
                double nabla_ij_y = -1.0 * prefactor_j * density_grad[j][1] * (nabla_inside_1+nabla_inside_2);
                double nabla_ij_z = -1.0 * prefactor_j * density_grad[j][2] * (nabla_inside_1+nabla_inside_2);
                gsl_bspline_eval(phi_j,splineValue, bw);
                for (int m=0; m<nSplines; m++)
                {
                     double phim = gsl_vector_get(splineValue, m);
                     if (phim < 1e-20)
                         continue;
                     // For all three dimensions
                     // frameMatrix[m][3*i]     += p_value * phim * prefactor_j * increment_x;
                     // frameMatrix[m][3*i + 1] += p_value * phim * prefactor_j * increment_y;
                     // frameMatrix[m][3*i + 2] += p_value * phim * prefactor_j * increment_z;
                     frameMatrix[m][3*i]     += q_value * phim * prefactor_j * increment_x;
                     frameMatrix[m][3*i + 1] += q_value * phim * prefactor_j * increment_y;
                     frameMatrix[m][3*i + 2] += q_value * phim * prefactor_j * increment_z;
                     // frameMatrix[2*nSplines+m][3*i]     += density_square[i] * phim * prefactor_j * increment_x;
                     // frameMatrix[2*nSplines+m][3*i + 1] += density_square[i] * phim * prefactor_j * increment_y;
                     // frameMatrix[2*nSplines+m][3*i + 2] += density_square[i] * phim * prefactor_j * increment_z;
                }
                //std::cout << phi << " " << phi_j << std::endl;
                double p_derivative = 30.0*(phi_j-1)*(phi_j-1)*phi_j*phi_j;
                frameVector[i][0] += p_value * p_derivative * prefactor_j * increment_x;
                frameVector[i][1] += p_value * p_derivative * prefactor_j * increment_y;
                frameVector[i][2] += p_value * p_derivative * prefactor_j * increment_z;
		frameVector[i][0] += 2.0 * epsilon_value * nabla_ij_x;
                frameVector[i][1] += 2.0 * epsilon_value * nabla_ij_y;
                frameVector[i][2] += 2.0 * epsilon_value * nabla_ij_z;
            }
        }
        gsl_bspline_eval(phi,splineValue,bw);
	double i_nabla_inside_1 = deriv_prefactor_i * (density_grad[i][0]*density_grad[i][0]+density_grad[i][1]*density_grad[i][1]+density_grad[i][2]*density_grad[i][2]);
        double i_nabla_inside_2 = prefactor_i * density_sq_factor[i] ;
        double nabla_i_x = -1.0 *prefactor_i*density_grad[i][0] * (i_nabla_inside_1+i_nabla_inside_2);
        double nabla_i_y = -1.0 *prefactor_i*density_grad[i][1] * (i_nabla_inside_1+i_nabla_inside_2);
        double nabla_i_z = -1.0 *prefactor_i*density_grad[i][2] * (i_nabla_inside_1+i_nabla_inside_2);
        for (int m=0; m<nSplines; m++)
        {
             double phim = gsl_vector_get(splineValue, m);
             if (phim < 1e-20)
                 continue;
             // For all three dimensions
             // FrameMatrix[m][3*i]     += p_value * phim * prefactor_i * grad_x;
             // FrameMatrix[m][3*i + 1] += p_value * phim * prefactor_i * grad_y;
             // FrameMatrix[m][3*i + 2] += p_value * phim * prefactor_i * grad_z;
             frameMatrix[m][3*i]     += q_value * phim * prefactor_i * grad_x;
             frameMatrix[m][3*i + 1] += q_value * phim * prefactor_i * grad_y;
             frameMatrix[m][3*i + 2] += q_value * phim * prefactor_i * grad_z;
             // frameMatrix[2*nSplines+m][3*i]     += density_square[i] * phim * prefactor_i * grad_x;
             // frameMatrix[2*nSplines+m][3*i + 1] += density_square[i] * phim * prefactor_i * grad_y;
             // frameMatrix[2*nSplines+m][3*i + 2] += density_square[i] * phim * prefactor_i * grad_z;
        }
        double p_derivative = 30.0*(phi-1)*(phi-1)*phi*phi;
        frameVector[i][0] += p_value * p_derivative * prefactor_i * grad_x;
        frameVector[i][1] += p_value * p_derivative * prefactor_i * grad_y;
        frameVector[i][2] += p_value * p_derivative * prefactor_i * grad_z;
	frameVector[i][0] += 2.0 * epsilon_value * nabla_i_x;
        frameVector[i][1] += 2.0 * epsilon_value * nabla_i_y;
        frameVector[i][2] += 2.0 * epsilon_value * nabla_i_z;
    }

    // Constructing the normal Matrix and normal Vector
    for (int m=0; m<nSplines; m++)
    {
        for (int n=0; n<nSplines; n++)
        {
            double sum = 0.0;
            for (int k=0; k<3 * nall; k++)
                sum += frameMatrix[m][k] * frameMatrix[n][k];
            normalMatrix[m][n] += sum * normalFactor;
        }
   
        double sum_b = 0.0; 
        for (int k=0; k<3 * nall; k++)
            sum_b += frameMatrix[m][k] * (trajFrame->residualForces[k/3][k%3]-frameVector[k/3][k%3]);
        normalVector[m] += sum_b * normalFactor;
    }
}

void FitGLE::leastSquareSolver()
{
    // Solving the least square normal equation G*phi = b
    int basisSize = info->numSplines;
    double* G = new double[basisSize * basisSize];
    double* b = new double[basisSize];

    /*
    std::vector<double> h(basisSize, 0.0);
    for (int i = 0; i < basisSize; i++) {
        for (int j = 0; j < basisSize; j++) {
            h[j] = h[j] + normalMatrix[i][j] * normalMatrix[i][j];  //mat->dense_fm_matrix[j * mat->accumulation_matrix_rows + i] * mat->dense_fm_matrix[j * mat->accumulation_matrix_rows + i];
        }
    }
    for (int i = 0; i < basisSize; i++) {
        if (h[i] < 1.0E-20) h[i] = 1.0;
        else h[i] = 1.0 / sqrt(h[i]);
    }
    for (int i =0; i < basisSize; i++)
    {
        for (int j=0; j < basisSize; j++)
           normalMatrix[i][j] *= h[j];
    }
*/

    // Store the normalMatrix in container 
    for (int m=0; m<basisSize; m++)
    {
        for (int n=0; n<basisSize; n++)
        {
            G[m*basisSize + n] = normalMatrix[m][n];
        }
        b[m] = normalVector[m];
        printf("m %d %lf\n", m, b[m]);
    }
    

    // Solving the linear system using SVD

    int m = basisSize;
    int n = basisSize;
    int nrhs = 1;
    int lda = basisSize;
    int ldb = 1;
    double rcond = -1.0;
    int irank;
    double* singularValue = new double[basisSize];
    int solverInfo = LAPACKE_dgelss(LAPACK_ROW_MAJOR, m, n, nrhs, G, lda, b, ldb, singularValue, rcond, &irank); 
   
    printf("LSQ Solver Info: %d\n", solverInfo);

    for (int m=0; m<basisSize; m++)
        splineCoefficients[m] = b[m];

    delete[] G;
    delete[] b;
}

// Output function for gamma(R)
void FitGLE::output()
{
    printf("output\n");
    double start = info->start;
    double end = info->end;
    double precision = info->outputPrecision;

    FILE* fb = fopen("spline_coeff.dat", "w");
    for (int m=0; m<info->numSplines; m++)
    {
        fprintf(fb, "%lf\n", splineCoefficients[m]);
    }
    fclose(fb);

    FILE* fp4 = fopen("q_phi.dat", "w");
    //FILE* fp2 = fopen("fw.dat", "w");
    //FILE* fp3 = fopen("fepsilon.dat", "w");

    while (start < end)
    {
        double gamma_r = 0.0;
        //double fp2_r = 0.0;
        //double fp3_r = 0.0;
        gsl_bspline_eval(start, splineValue, bw);
        for (int m=0; m<info->numSplines; m++)
        {
           gamma_r += splineCoefficients[m] * gsl_vector_get(splineValue, m);
        //   fp2_r += splineCoefficients[m+1*info->numSplines] * gsl_vector_get(splineValue, m);
        //   fp3_r += splineCoefficients[m+2*info->numSplines] * gsl_vector_get(splineValue, m);
        }
        fprintf(fp4, "%lf\t%lf\n", start, gamma_r);
        //fprintf(fp2, "%lf\t%lf\n", start, fp2_r);
        //fprintf(fp3, "%lf\t%lf\n", start, fp3_r);
        start = start + precision;
    }
    fclose(fp4);
    //fclose(fp2);
    //fclose(fp3);
}

// Execution Process
void FitGLE::exec()
{
    printf("Accumulating the LSQ normal Matrix\n");
    for (int i=0; i<info->steps; i++)
    {
        trajFrame->readFrame();
        accumulateNormalEquation();
        printf("finishing step %d (total %d)\r", i+1, info->steps);
    }
    printf("\n");
    leastSquareSolver();
    output();
}
