#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <iostream>

#include "../lib/data.hpp"

std::vector<std::vector<double>> read_csv(std::string path) {
    std::vector<std::vector<double>> dat;
    std::ifstream file(path);

    if(file.is_open()) {
        std::string line, val;
        std::getline(file, line);

        unsigned int columns = 1;
        for(char &ch: line)
            columns += (ch == ',');

        dat.resize(columns, std::vector<double>());
        while(std::getline(file, line)) {
            for(unsigned int col = 0; col < columns; col++) {
                double val = std::stod(line.substr(0, line.find(",")));
                dat[col].push_back(val);

                line = line.substr(line.find(",") + 1);
            }
        }

        file.close();
    }

    return dat;
}

void standardize(std::vector<double> &dat) {
    double mean = 0.00;
    for(double &val: dat)
        mean += val;
    mean /= dat.size();

    double std_dev = 0.00;
    for(double &val: dat)
        std_dev += pow(val - mean, 2);
    std_dev /= dat.size();
    std_dev = sqrt(std_dev);

    for(double &val: dat)
        val = (val - mean) / std_dev;
}

std::string Market::ticker(unsigned int i) {
    return tickers[i];
}

unsigned int Market::num_of_assets() {
    return assets.size();
}

std::vector<double> *Market::asset(unsigned int i) {
    return &assets[i];
}

std::vector<double> *Memory::state() {
    return &s;
}

unsigned int Memory::action() {
    return a;
}

double Memory::expected_reward() {
    return r;
}