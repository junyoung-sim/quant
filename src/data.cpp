
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

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

std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int period) {
    std::vector<double> weight;
    double weight_sum = 0.00;
    double smoothing = 2.00 / (period + 1);
    for(unsigned int i = 0; i < period; i++) {
        double w = pow(1.00 - smoothing, period - 1 - i);
        weight.push_back(w);
        weight_sum += w;
    }

    std::vector<double> ema;
    for(unsigned int t = 0; t <= series.size() - period; t++) {
        double exp_sum = 0.00;
        for(unsigned int i = t; i < t + period; i++)
            exp_sum += series[i] * weight[i-t];
        ema.push_back(exp_sum / weight_sum);
    }

    std::vector<double>().swap(weight);

    return ema;
}

// --- //

unsigned int Market::num_of_assets() {
    return assets.size();
}

std::string Market::ticker(unsigned int i) {
    return tickers[i];
}

std::vector<double> *Market::asset(unsigned int i) {
    return &assets[i];
}
