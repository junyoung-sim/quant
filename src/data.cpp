
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

std::vector<double> simple_moving_average(std::vector<double> &series, unsigned int period) {
    std::vector<double> sma;
    for(unsigned int t = 0; t <= series.size() - period; t++) {
        double sum = 0.00;
        for(unsigned int i = t; i < t + period; i++)
            sum += series[i];
        sma.push_back(sum / period);
    }

    return sma;
}

std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int period) {
    std::vector<double> weights;
    double weight_sum = 0.00;
    double smoothing = 2.00 / (period + 1);
    for(unsigned int t = 0; t < period; t++) {
        double weight = pow(1.00 - smoothing, period - 1 - t);
        weights.push_back(weight);
        weight_sum += weight;
    }

    std::vector<double> ema;
    for(unsigned int t = 0; t <= series.size() - period; t++) {
        double sum = 0.00;
        for(unsigned int i = t; i < t + period; i++)
            sum += series[i] * weights[i-t];
        ema.push_back(sum / weight_sum);
    }

    std::vector<double>().swap(weights);

    return ema;
}

std::vector<double> moving_expectation_value(std::vector<double> &series, unsigned int period) {
    std::vector<double> mev;
    for(unsigned int t = 1; t <= series.size() - period; t++) {
        unsigned int gain_count = 0, loss_count = 0;
        double mean_gain = 0.00, mean_loss = 0.00;
        double gain_prob = 0.00, loss_prob = 0.00;
        double expectation = 0.00;

        for(unsigned int i = t; i < t + period; i++) {
            double diff = (series[i] - series[i-1]) / series[i-1];
            if(diff > 0.00) {
                mean_gain += diff;
                gain_count++;
            }
            else {
                mean_loss += diff;
                loss_count++;
            }
        }

        if(gain_count == 0)
            mean_gain = 0.00;
        else
            mean_gain /= gain_count;

        if(loss_count == 0)
            mean_loss = 0.00;
        else
            mean_loss /= loss_count;

        gain_prob = (double)gain_count / (gain_count + loss_count);
        loss_prob = (double)loss_count / (gain_count + loss_count);

        expectation = gain_prob * mean_gain + loss_prob * mean_loss;

        mev.push_back(expectation);
    }

    return mev;
}

// --- //

std::vector<double> sample_state(std::vector<double> &series, unsigned int t, unsigned int look_back) {
    std::vector<double> price = {series.begin() + t - look_back + 1, series.begin() + t + 1};
    range_normalize(price);

    double short_term_change = (price[price.size() - 1] - price[price.size() - 10]) / price[price.size() - 10];
    double mid_term_change = (price[price.size() - 1] - price[price.size() - 50]) / price[price.size() - 50];
    double long_term_change = (price[price.size() - 1] - price[price.size() - 100]) / price[price.size() - 100];

    std::vector<double> fast_ema = exponential_moving_average(price, 10);
    std::vector<double> slow_ema = exponential_moving_average(price, 50);

    std::vector<double> mev = moving_expectation_value(price, 20);
    std::vector<double> mev_sma = simple_moving_average(mev, 10);

    std::vector<double> state;
    state.push_back(price[price.size() - 1]);
    state.push_back(short_term_change);
    state.push_back(mid_term_change);
    state.push_back(long_term_change);
    state.push_back(fast_ema[fast_ema.size() - 1]);
    state.push_back(slow_ema[slow_ema.size() - 1]);
    state.push_back(mev[mev.size() - 1]);
    state.push_back(mev_sma[mev_sma.size() - 1]);

    std::vector<double>().swap(price);
    std::vector<double>().swap(fast_ema);
    std::vector<double>().swap(slow_ema);
    std::vector<double>().swap(mev);
    std::vector<double>().swap(mev_sma);

    return state;
}
