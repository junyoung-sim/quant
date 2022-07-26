
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

std::vector<double> read_csv(std::string path, std::string column) {
    std::ifstream source(path);
    std::vector<double> series;

    if(source.is_open()) {
        std::string header;
        std::getline(source, header);

        unsigned int delimiter_count = 0;
        for(unsigned int i = 0; i < static_cast<unsigned int>(header.find(column)); i++) {
            if(header[i] == ',')
                delimiter_count += 1;
        }

        std::string line, val;
        while(std::getline(source, line)) {
            std::string val;
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
        }

        source.close();
    }

    return series;
}

void write(std::string path, std::vector<double> &series) {
    std::ofstream out(path);

    if(out.is_open()) {
        for(unsigned int i = 0; i < series.size(); i++) {
            out << series[i];
            if(i != series.size() - 1)
                out << " ";
        }

        out.close();
    }
}

void standardize(std::vector<double> &series) {
    double mean = 0.00;
    for(double &val: series)
        mean += val;
    mean /= series.size();

    double std = 0.00;
    for(double &val: series)
        std += pow(val - mean, 2);
    std /= series.size();
    std = sqrt(std);

    for(double &val: series)
        val = (val - mean) / std;
}

std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int periods) {
    std::vector<double> weights;
    double weight_count = 0.00;
    double smoothing = 2.00 / (periods + 1);
    for(unsigned int t = 0; t < periods; t++) {
        weights.push_back(pow(1.00 - smoothing, periods - 1 - t));
        weight_count += weights[weights.size() - 1];
    }

    std::vector<double> ema;
    for(unsigned int t = 0; t <= series.size() - periods; t++) {
        double weighted_sum = 0.00;
        for(unsigned int i = t; i < t + periods; i++)
            weighted_sum += series[i] * weights[i-t];

        ema.push_back(weighted_sum / weight_count);
    }

    return ema;
}


