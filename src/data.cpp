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
        for(unsigned int i = 0; i < static_cast<unsigned int>(header.find(column)); i++)
            delimiter_count += (header[i] == ',');

        std::string line, val;
        while(std::getline(source, line)) {
            unsigned int dc = 0;
            for(char &ch: line) {
                if(dc != delimiter_count)
                    dc += (ch == ',');
                else {
                    if(ch != ',')
                        val += ch;
                    else
                        break;
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