
#include <cstdlib>
#include <vector>
#include <string>

std::vector<double> read_csv(std::string path, std::string column);

std::vector<double> standardize(std::vector<double> &series);
std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int periods);
std::vector<double> moving_average_convergence_divergence(std::vector<double> &series);
std::vector<double> stochastic_oscillator(std::vector<double> &series, unsigned int periods);
std::vector<double> relative_strength_index(std::vector<double> &series, unsigned int periods);
