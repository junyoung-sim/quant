#ifndef __DATA_HPP_
#define __DATA_HPP_

#include <cstdlib>
#include <vector>
#include <string>

std::vector<double> read_csv(std::string path, std::string column);

// --- //

void range_normalize(std::vector<double> &series);
void standardize(std::vector<double> &series);

std::vector<double> simple_moving_average(std::vector<double> &series, unsigned int period);
std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int period);

// --- //

std::vector<double> sample_state(std::vector<double> &series, unsigned int t, unsigned int look_back);

#endif