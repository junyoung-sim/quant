
#include <cstdlib>
#include <vector>
#include <string>

std::vector<double> read_csv(std::string path, std::string column);
void write(std::string path, std::vector<double> &series);

void standardize(std::vector<double> &series);
void range_normalize(std::vector<double> &series);

std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int periods);
