
#include <cstdlib>
#include <vector>
#include <string>

std::vector<double> read_csv(std::string path, std::string column);
void write(std::string path, std::vector<double> &series);

void standardize(std::vector<double> &series);

std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int periods);
std::vector<double> moving_average_convergence_divergence(std::vector<double> &series, unsigned int fast_period, unsigned int slow_period);
std::vector<double> stochastic_oscillator(std::vector<double> &series, unsigned int k_period, unsigned int d_period);
