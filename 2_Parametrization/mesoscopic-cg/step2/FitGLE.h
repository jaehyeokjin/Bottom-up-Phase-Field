#ifndef _FITGLE_H_
#define _FITGLE_H_

#include <cstdlib>
#include <cstdio>
#include <vector>
#include "Frame.h"
#include <memory>
#include <gsl/gsl_bspline.h>

namespace FITGLE_NS {

struct InputParameters
{
    double start;  //start distance r0
    double end;    //end distance r1
    int    splineOrder;
    int    numSplines;
    double outputPrecision;
    double boxLengthX; 
    double boxLengthY; 
    double boxLengthZ; 
    int    steps;
    FILE*  fileTraj;
};  //Structure to store input parameters

class FitGLE
{
public:
    FitGLE(int argc, char** argv);
    ~FitGLE();
    void exec();

    //helper functions
    void accumulateNormalEquation();
    void leastSquareSolver();
    void output();
    
    //helper functions
    double distance(std::vector<double> &, std::vector<double> &);
    std::vector<double> parallelVelocity(int i, int j);
    std::vector<double> parallelUnitVector(int i, int j);

private:
    std::shared_ptr<class Frame> trajFrame;
    //class Frame* trajFrame;
    std::shared_ptr<struct InputParameters> info; 
    std::vector<double> divPoints;  // divide points of the b-spline radial ranges
    std::vector<std::vector<double> > normalMatrix;
    std::vector<double> normalVector;
    std::vector<double> splineCoefficients;
    std::vector<double> density;
    std::vector<double> density_square;
    std::vector<double> density_sq_factor;

    //gsl members for b-splines
    gsl_bspline_workspace *bw;
    gsl_vector *splineValue;

};

}

#endif
