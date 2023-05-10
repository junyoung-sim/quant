#ifndef __DATA_HPP_
#define __DATA_HPP_

#include <vector>
#include <string>

void download(std::vector<std::string> &tickers);

std::vector<std::vector<double>> read_csv(std::string path);

std::vector<std::vector<double>> historical_data(std::string ticker, std::vector<std::string> &indicators);

double mean(std::vector<double> &dat);
double stdev(std::vector<double> &dat);

void piecewise_aggregate_approximation(std::vector<double> &dat, unsigned int window);

void standardize(std::vector<double> &dat);

#endif