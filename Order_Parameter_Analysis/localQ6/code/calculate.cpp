#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <complex>
#include <string>
#include <iomanip>
#include <algorithm>

// Boost provides spherical harmonic functions
#include <boost/math/special_functions/spherical_harmonic.hpp>

// New switching function parameters (rational switching function)
// d0 = 0.0, n = 6, m = 12, r0 = 3.5 (R_0)
const double d0 = 0.0;
const double r0 = 3.5;
const double n_param = 6;
const double m_param = 12;

// Rational switching function: 
// s(r) = (1 - (r/r0)^n) / (1 - (r/r0)^m) for r < r0, and 0 otherwise.
double sigma(double r) {
    if (r >= r0)
        return 0.0;
    double ratio = (r - d0) / r0;  // equals r / r0, since d0 is 0.
    double numerator = 1.0 - std::pow(ratio, n_param);
    double denominator = 1.0 - std::pow(ratio, m_param);
    return numerator / denominator;
}

// Apply minimum image convention for a displacement component.
double minimumImage(double dx, double boxLength) {
    if (dx > 0.5 * boxLength)
        dx -= boxLength;
    else if (dx < -0.5 * boxLength)
        dx += boxLength;
    return dx;
}

// Structure to hold particle information.
struct Particle {
    int id;
    int type;
    double x, y, z;
};

// Structure for box dimensions.
struct Box {
    double xlo, xhi;
    double ylo, yhi;
    double zlo, zhi;
};

// Compute Q6 for a given particle i using its neighbors, considering periodic boundaries.
double computeQ6(const Particle &pi, const std::vector<Particle> &particles, const Box &box) {
    // Sum over m = -6 ... +6 (13 components).
    std::complex<double> q6[13] = {0.0};
    double weightSum = 0.0;

    for (const auto &pj : particles) {
        if (pi.id == pj.id)
            continue;
        // Compute displacement using minimum image convention.
        double dx = pj.x - pi.x;
        double dy = pj.y - pi.y;
        double dz = pj.z - pi.z;
        double Lx = box.xhi - box.xlo;
        double Ly = box.yhi - box.ylo;
        double Lz = box.zhi - box.zlo;
        dx = minimumImage(dx, Lx);
        dy = minimumImage(dy, Ly);
        dz = minimumImage(dz, Lz);

        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        double w = sigma(r);
        if (w > 0.0 && r > 1e-12) {
            // Convert displacement to spherical coordinates.
            double theta = std::acos(dz / r);
            double phi = std::atan2(dy, dx);
            // Sum contributions for m from -6 to 6.
            for (int m = -6; m <= 6; m++) {
                // Get the real spherical harmonic value.
                double Y_real = boost::math::spherical_harmonic_r(6, m, theta, phi);
                // Include phase factor to form a complex spherical harmonic.
                std::complex<double> Y = std::polar(Y_real, m * phi);
                q6[m + 6] += w * Y;
            }
            weightSum += w;
        }
    }

    double q6_norm = 0.0;
    if (weightSum > 1e-12) {
        for (int m = 0; m < 13; m++) {
            std::complex<double> q6m = q6[m] / weightSum;
            q6_norm += std::norm(q6m);
        }
        q6_norm = std::sqrt(q6_norm);
    }
    return q6_norm;
}

// Create a histogram from a vector of Q6 values.
// It returns a vector of counts for each bin and also sets minVal and maxVal.
std::vector<int> createHistogram(const std::vector<double> &q6Values, int numBins, double &minVal, double &maxVal) {
    if(q6Values.empty()) return std::vector<int>(numBins, 0);
    
    minVal = *std::min_element(q6Values.begin(), q6Values.end());
    maxVal = *std::max_element(q6Values.begin(), q6Values.end());
    // Expand the range if necessary.
    if (fabs(maxVal - minVal) < 1e-6) {
        maxVal += 1e-3;
        minVal -= 1e-3;
    }
    std::vector<int> histogram(numBins, 0);
    double binWidth = (maxVal - minVal) / numBins;
    for (auto val : q6Values) {
        int bin = static_cast<int>((val - minVal) / binWidth);
        if (bin == numBins) bin = numBins - 1;
        histogram[bin]++;
    }
    return histogram;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./compute_q6 <trajectory_file>\n";
        return 1;
    }
    
    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }
    
    std::string line;
    int timestep;
    int nAtoms;
    Box box;
    std::vector<Particle> particles;
    
    // We'll accumulate all per-particle Q6 values over all timesteps for the histogram.
    std::vector<double> allQ6Values;
    // We'll also store the average Q6 for each timestep.
    std::vector<double> avgQ6PerTimestep;
    int nTimesteps = 0;
    
    // Process file timestep by timestep.
    while (std::getline(infile, line)) {
        if (line.find("ITEM: TIMESTEP") != std::string::npos) {
            nTimesteps++;
            // Read timestep.
            std::getline(infile, line);
            std::istringstream tsStream(line);
            tsStream >> timestep;
            
            // Read number of atoms.
            std::getline(infile, line); // ITEM: NUMBER OF ATOMS
            std::getline(infile, line);
            std::istringstream natomStream(line);
            natomStream >> nAtoms;
            
            // Read box bounds.
            std::getline(infile, line); // ITEM: BOX BOUNDS pp pp pp
            std::getline(infile, line);
            std::istringstream boxStream1(line);
            boxStream1 >> box.xlo >> box.xhi;
            std::getline(infile, line);
            std::istringstream boxStream2(line);
            boxStream2 >> box.ylo >> box.yhi;
            std::getline(infile, line);
            std::istringstream boxStream3(line);
            boxStream3 >> box.zlo >> box.zhi;
            
            // Read header for atoms.
            std::getline(infile, line); // ITEM: ATOMS id type x y z fx fy fz
            
            // Read atoms data.
            particles.clear();
            particles.reserve(nAtoms);
            for (int i = 0; i < nAtoms; i++) {
                if (!std::getline(infile, line)) break;
                std::istringstream iss(line);
                Particle p;
                iss >> p.id >> p.type >> p.x >> p.y >> p.z;
                particles.push_back(p);
            }
            
            // For each particle, compute Q6 and add to our accumulated list.
            double q6Sum = 0.0;
            int count = 0;
            for (const auto &p : particles) {
                double q6_val = computeQ6(p, particles, box);
                allQ6Values.push_back(q6_val);
                q6Sum += q6_val;
                count++;
            }
            // Calculate and store average Q6 for this timestep.
            double avgQ6 = (count > 0) ? (q6Sum / count) : 0.0;
            avgQ6PerTimestep.push_back(avgQ6);
        }
    }
    
    infile.close();
    
    // Create the final histogram from all per-particle Q6 values over the whole trajectory.
    int numBins = 50;
    double minVal, maxVal;
    std::vector<int> histogram = createHistogram(allQ6Values, numBins, minVal, maxVal);
    
    // Write the final histogram to an output file.
    std::ofstream histOut("final_histogram.txt");
    if (!histOut) {
        std::cerr << "Error: Cannot open output file for final histogram\n";
        return 1;
    }
    
    histOut << "Total timesteps processed: " << nTimesteps << "\n";
    histOut << "Histogram of per-particle Q6 values (over whole trajectory):\n";
    histOut << "Bin_Range\tCount\n";
    double binWidth = (maxVal - minVal) / numBins;
    for (int i = 0; i < numBins; i++) {
        double binStart = minVal + i * binWidth;
        double binEnd = binStart + binWidth;
        histOut << std::fixed << std::setprecision(4)
                << "[" << binStart << ", " << binEnd << "):\t" << histogram[i] << "\n";
    }
    histOut.close();
    
    // Write the average Q6 per timestep to a separate file.
    std::ofstream avgOut("q6_avg_per_timestep.txt");
    if (!avgOut) {
        std::cerr << "Error: Cannot open output file for Q6 average per timestep\n";
        return 1;
    }
    avgOut << "Timestep\tAverage_Q6\n";
    for (size_t i = 0; i < avgQ6PerTimestep.size(); i++) {
        avgOut << (i+1) << "\t" << std::fixed << std::setprecision(6) << avgQ6PerTimestep[i] << "\n";
    }
    avgOut.close();
    
    std::cout << "Processing complete. Total timesteps processed: " << nTimesteps << "\n";
    std::cout << "Final histogram written to final_histogram.txt\n";
    std::cout << "Average Q6 per timestep written to q6_avg_per_timestep.txt\n";
    
    return 0;
}

