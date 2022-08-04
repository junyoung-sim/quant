
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
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

unsigned int Quant::policy(std::vector<double> &state) {
    std::vector<double> agent_q = agent.predict(state);
    unsigned int action = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();

    std::vector<double>().swap(agent_q);

    return action;
}

unsigned int Quant::eps_greedy_policy(std::vector<double> &state, double eps) {
    unsigned int action;
    double explore = (double)rand() / RAND_MAX;

    if(explore < eps)
        action = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
    else
        action = policy(state);

    return action;
}

void Quant::optimize(double eps_init, double eps_min, double alpha_init, double alpha_min, 
                     double gamma, unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval) {
    double eps = eps_init;
    double alpha = alpha_min;

    std::vector<Memory> memory;

    std::vector<double> *main_asset = market->asset(MAIN_ASSET);

    double loss_sum = 0.00, mean_loss = 0.00;
    double benchmark = 1.00, model = 1.00;

    unsigned int start = (moving_average_period - 1) + (look_back - 1);
    unsigned int terminal = main_asset->size() - 2;
    unsigned int training_count = 0;

    for(unsigned int t = start; t <= terminal; t++) {
        std::vector<double> state = sample_state(t);
        unsigned int action = eps_greedy_policy(state, eps);

        double diff = (main_asset->at(t+1) - main_asset->at(t)) / main_asset->at(t);
        double observed_reward;
        if(action == LONG)
            observed_reward = diff;
        else if(action == SHORT)
            observed_reward = -diff;
        else
            observed_reward = 0.00;

        double expected_reward = observed_reward;
        if(t != terminal) {
            std::vector<double> next_state = sample_state(t+1);
            std::vector<double> target_q = target.predict(next_state);
            expected_reward += gamma * *std::max_element(target_q.begin(), target_q.end());

            std::vector<double>().swap(next_state);
            std::vector<double>().swap(target_q);
        }

        if(training_count > 0) {
            std::vector<double> agent_q = agent.predict(state);
            loss_sum += pow(expected_reward - agent_q[action], 2);
            mean_loss = loss_sum / training_count;

            benchmark *= 1.00 + diff;
            model *= 1.00 + observed_reward;

            std::ofstream log("./res/log", std::ios::app);
            log << mean_loss << " " << benchmark << " " << model << "\n";
            log.close();

            std::vector<double>().swap(agent_q);
        }

        memory.push_back(Memory(state, action, expected_reward));
        std::vector<double>().swap(state);

        std::cout << "@frame" << t << ": action = " << action << " -> observed = " << observed_reward << ", expected = " << expected_reward << " ";
        std::cout << "(benchmark = " << benchmark << ", model = " << model << ")\n";
    }

    std::vector<Memory>().swap(memory);
}