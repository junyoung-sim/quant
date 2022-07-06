
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

std::vector<double> moving_average_convergence_divergence(std::vector<double> &series) {
    std::vector<double> ema12 = exponential_moving_average(series, 12);
    std::vector<double> ema26 = exponential_moving_average(series, 26);
    ema12.erase(ema12.begin(), ema12.begin() + (ema12.size() - ema26.size()));

    std::vector<double> macd;
    for(unsigned int t = 0; t < ema12.size(); t++)
        macd.push_back(ema12[t] - ema26[t]);

    return macd;
}

std::vector<double> stochastic_oscillator(std::vector<double> &series, unsigned int periods) {
    std::vector<double> osc;
    for(unsigned int t = 0; t <= series.size() - periods; t++) {
        std::vector<double> series_t = {series.begin() + t, series.begin() + t + periods};
        double max = *std::max_element(series_t.begin(), series_t.end());
        double min = *std::min_element(series_t.begin(), series_t.end());

        osc.push_back((series_t[periods-1] - min) / (max - min));
    }

    return osc;
}

std::vector<double> relative_strength_index(std::vector<double> &series, unsigned int periods) {
    std::vector<double> rsi;
    for(unsigned int t = 0; t <= series.size() - periods; t++) {
        double mean_gain = 0.00, mean_loss = 0.00;
        for(unsigned int i = t; i < t + periods - 1; i++) {
            double delta = (series[i+1] - series[i]) / series[i];
            if(delta > 0.00)
                mean_gain += delta;
            else
                mean_loss += abs(delta);
        }

        if(mean_loss == 0.00)
            rsi.push_back(1.00);
        else
            rsi.push_back(1.00 - 1.00 / (1.00 + mean_gain / mean_loss));
    }

    return rsi;
}

