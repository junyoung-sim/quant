
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>

#include "../lib/quant.hpp"

void Quant::init(std::vector<std::vector<unsigned int>> shape) {
    srand(time(NULL));
    double kernel_max = 1.0, kernel_min = -1.0;
    for(unsigned int i = 0; i < market->num_of_assets(); i++) {
        double kernel_val = kernel_min + (double)rand() * (kernel_max - kernel_min) / RAND_MAX;
        kernel.push_back(kernel_val);
    }

    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }

    seed.seed(std::chrono::system_clock::now().time_since_epoch().count());
    agent.init(seed);
    sync();
}

void Quant::sync() {
    for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                double weight = agent.layer(l)->node(n)->weight(i);
                target.layer(l)->node(n)->set_weight(i, weight);
            }

            double bias = agent.layer(l)->node(n)->bias();
            target.layer(l)->node(n)->set_bias(bias);
        }
    }
}

std::vector<double> Quant::sample_state(unsigned int t) {
    std::vector<std::vector<double>> state2d;
    for(unsigned int i = 0; i < market->num_of_assets(); i++) {
        std::vector<double> *asset = market->asset(i);
        std::vector<double> asset_t = {asset->begin(), asset->begin() + t + 1};

        std::vector<double> ema = exponential_moving_average(asset_t, moving_average_period);
        ema.erase(ema.begin(), ema.end() - look_back);
        standardize(ema);

        state2d.push_back(ema);

        std::vector<double>().swap(asset_t);
        std::vector<double>().swap(ema);
    }

    std::vector<double> state1d;
    for(unsigned int j = 0; j < state2d[0].size(); j++) {
        double dot = 0.00;
        for(unsigned int i = 0; i < state2d.size(); i++)
            dot += state2d[i][j] * kernel[i];
        state1d.push_back(relu(dot));
    }

    return state1d;
}
