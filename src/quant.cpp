
#include <cstdlib>
#include <vector>

#include "../lib/quant.hpp"

void Quant::init(std::vector<std::vector<unsigned int>> shape) {
    for(unsigned int i = 0; i < market->num_of_assets(); i++)
        kernel.push_back((double)rand() / RAND_MAX);

    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }

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
        std::vector<double> ema = exponential_moving_average(*market->asset(i), moving_average_period);
        ema.erase(ema.begin(), ema.end() - look_back);
        standardize(ema);

        state2d.push_back(ema);
        std::vector<double>().swap(ema);
    }

    std::vector<double> state1d;
    for(unsigned int t = 0; t < state2d[0].size(); t++) {
        double dot = 0.00;
        for(unsigned int i = 0; i < state2d.size(); i++)
            dot += state2d[i][t] * kernel[i];
        state1d.push_back(relu(dot));
    }

    return state1d;
}
