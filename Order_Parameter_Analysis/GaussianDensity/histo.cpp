#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>

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

// Apply minimum image convention for a displacement component.
double minimumImage(double dx, double boxLength) {
    if (dx > 0.5 * boxLength)
        dx -= boxLength;
    else if (dx < -0.5 * boxLength)
        dx += boxLength;
    return dx;
}

// Compute the new local order parameter phi_I for particle i:
//   rho_I = sum_{J!=I} exp(-r^2/793)
//   phi_I = 0.5*(1 - tanh((rho_I - 1126.0)/2.0))
double computePhi(
    const Particle &pi,
    const std::vector<Particle> &particles,
    const Box &box
) {
    double Lx = box.xhi - box.xlo;
    double Ly = box.yhi - box.ylo;
    double Lz = box.zhi - box.zlo;
    double rho = 0.0;

    for (const auto &pj : particles) {
        if (pj.id == pi.id) continue;

        double dx = minimumImage(pj.x - pi.x, Lx);
        double dy = minimumImage(pj.y - pi.y, Ly);
        double dz = minimumImage(pj.z - pi.z, Lz);

        double rsq = dx*dx + dy*dy + dz*dz;
        rho += std::exp(-rsq / 793.0);
    }

    return 0.5 * (1.0 - std::tanh((rho - 1126.0) / 5.0));
}

// Create a histogram from a vector of phi values.
// Returns counts per bin and sets minVal and maxVal.
std::vector<int> createHistogram(
    const std::vector<double> &values,
    int numBins,
    double &minVal,
    double &maxVal
) {
    if (values.empty())
        return std::vector<int>(numBins, 0);

    minVal = *std::min_element(values.begin(), values.end());
    maxVal = *std::max_element(values.begin(), values.end());
    if (std::fabs(maxVal - minVal) < 1e-6) {
        maxVal += 1e-3;
        minVal -= 1e-3;
    }

    std::vector<int> histogram(numBins, 0);
    double binWidth = (maxVal - minVal) / numBins;

    for (double v : values) {
        int bin = static_cast<int>((v - minVal) / binWidth);
        if (bin < 0) bin = 0;
        if (bin >= numBins) bin = numBins - 1;
        histogram[bin]++;
    }
    return histogram;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./compute_phi_hist <trajectory_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    std::string line;
    int timestep, nAtoms;
    Box box;
    std::vector<Particle> particles;

    // Accumulate per-particle phi values over all timesteps
    std::vector<double> allPhiValues;
    // Store average phi per timestep
    std::vector<double> avgPhiPerTimestep;
    int nTimesteps = 0;

    while (std::getline(infile, line)) {
        if (line.find("ITEM: TIMESTEP") != std::string::npos) {
            ++nTimesteps;
            // Read timestep
            std::getline(infile, line);
            std::istringstream(line) >> timestep;

            // Read number of atoms
            std::getline(infile, line);
            std::getline(infile, line);
            std::istringstream(line) >> nAtoms;

            // Read box bounds
            std::getline(infile, line);
            std::getline(infile, line);
            std::istringstream(line) >> box.xlo >> box.xhi;
            std::getline(infile, line);
            std::istringstream(line) >> box.ylo >> box.yhi;
            std::getline(infile, line);
            std::istringstream(line) >> box.zlo >> box.zhi;

            // Read atom header
            std::getline(infile, line);

            // Read atom data
            particles.clear();
            particles.reserve(nAtoms);
            for (int i = 0; i < nAtoms; ++i) {
                std::getline(infile, line);
                std::istringstream iss(line);
                Particle p;
                iss >> p.id >> p.type >> p.x >> p.y >> p.z;
                particles.push_back(p);
            }

            // Compute phi for each particle
            double phiSum = 0.0;
            for (const auto &p : particles) {
                double phi = computePhi(p, particles, box);
                allPhiValues.push_back(phi);
                phiSum += phi;
            }
            double avgPhi = particles.empty() ? 0.0 : (phiSum / particles.size());
            avgPhiPerTimestep.push_back(avgPhi);
        }
    }
    infile.close();

    // Make histogram
    int numBins = 50;
    double minVal, maxVal;
    auto histogram = createHistogram(allPhiValues, numBins, minVal, maxVal);

    // Write histogram file
    std::ofstream histOut("final_phi_histogram.txt");
    histOut << "Total timesteps processed: " << nTimesteps << "\n";
    histOut << "Histogram of per-particle phi values:\n";
    histOut << "Bin_Range\tCount\n";
    double binWidth = (maxVal - minVal) / numBins;
    for (int i = 0; i < numBins; ++i) {
        double start = minVal + i*binWidth;
        double end   = start + binWidth;
        histOut << std::fixed << std::setprecision(4)
                << "[ " << start << " ," << end << "):\t" << histogram[i] << "\n";
    }
    histOut.close();

    // Write average phi per timestep
    std::ofstream avgOut("phi_avg_per_timestep.txt");
    avgOut << "Timestep\tAverage_phi\n";
    for (size_t i = 0; i < avgPhiPerTimestep.size(); ++i) {
        avgOut << (i+1) << "\t" << std::fixed << std::setprecision(6)
               << avgPhiPerTimestep[i] << "\n";
    }
    avgOut.close();

    std::cout << "Processed " << nTimesteps << " timesteps.\n";
    std::cout << "Histogram written to final_phi_histogram.txt\n";
    std::cout << "Average phi per timestep written to phi_avg_per_timestep.txt\n";

    return 0;
}
