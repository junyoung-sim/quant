#ifndef __DATA_HPP_
#define __DATA_HPP_

#include <vector>
#include <string>

void download(std::vector<std::string> &tickers);

std::vector<std::vector<double>> read_csv(std::string path);

std::vector<std::vector<double>> historical_data(std::string ticker, std::vector<std::string> &indicators);

#endif