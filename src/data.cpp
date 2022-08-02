
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

#include <iostream>

#include "../lib/data.hpp"

std::vector<double> read_csv(std::string path, std::string column) {
    std::ifstream source(path);
    std::vector<double> series;

    if(source.is_open()) {
        std::string header;
        std::getline(source, header);

        unsigned int delimiter_count = 0;
        for(unsigned int i = 0; i < static_cast<unsigned int>(header.find(column)); i++) {
            if(header[i] == ',')
                delimiter_count++;
        }

        std::string line, val;
        while(std::getline(source, line)) {
            unsigned int dc = 0;
            for(unsigned int i = 0; i < line.length(); i++) {
                if(dc == delimiter_count) {
                    if(line[i] != ',')
                        val += line[i];
                    else
                        break;
                }
                else {
                    if(line[i] == ',')
                        dc++;
                }
            }

            series.push_back(std::stod(val));
            val = "";
        }

        source.close();
    }

    return series;
}

// --- //

void range_normalize(std::vector<double> &series) {
    double max = *std::max_element(series.begin(), series.end());
    double min = *std::min_element(series.begin(), series.end());

    for(double &val: series)
        val = (val - min) / (max - min);
}

void standardize(std::vector<double> &series) {
    double mean = 0.00;
    for(double &val: series)
        mean += val;
    mean /= series.size();

    double std_dev = 0.00;
    for(double &val: series)
        std_dev += pow(val - mean, 2);
    std_dev /= series.size();
    std_dev = sqrt(std_dev);

    for(double &val: series)
        val = (val - mean) / std_dev;
}

// --- //

std::vector<double> sample_state(std::vector<double> &series, unsigned int t) {
    std::vector<double> price = {series.begin() + t - 99, series.begin() + t + 1};

    std::vector<double> state;
    for(unsigned int t = 10; t <= price.size(); t += 10)
        state.push_back(price[t-1]);

    range_normalize(state);

    return state;
}
