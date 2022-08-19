
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iostream>

#include "../lib/quant.hpp"

void Quant::init(std::vector<std::vector<unsigned int>> shape) {
    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }

    srand(time(NULL));
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

// --- //

std::vector<double> Quant::sample_state(unsigned int market_id, unsigned int t) {
    std::vector<double> state;
    Market *market = &market_dataset->at(market_id);
    for(unsigned int i = 0; i < market->num_of_assets(); i++) {
        std::vector<double> *asset = market->asset(i);
        std::vector<double> asset_t = {asset->begin() + t + 1 - look_back, asset->begin() + t + 1};
        standardize(asset_t);

        state.insert(state.end(), asset_t.begin(), asset_t.end());

        std::vector<double>().swap(asset_t);
    }

    return state;
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

// --- //

void Quant::optimize() {
    double eps_init = 1.00;
    double eps_min = 0.10;
    double alpha_init = 0.0001;
    double alpha_min = 0.00001;
    double lambda = 0.05;
    double gamma = 0.50;
    unsigned int batch_size = 32;
    unsigned int memory_capacity = (unsigned int)(num_of_frames * 0.10);
    unsigned int sync_interval = 1000;

    std::vector<Memory> memory;

    double eps = eps_init;
    double alpha = alpha_init;
    unsigned int frame_count = 0;

    double loss_sum = 0.00, mean_loss = 0.00;

    for(unsigned int m = 0; m < market_dataset->size(); m++) {
        Market *market = &market_dataset->at(m);
        unsigned int start = look_back - 1;
        unsigned int terminal = market->asset(MAIN_ASSET)->size() - 2;

        double benchmark = 0.00, model = 0.00;

        for(unsigned int t = start; t <= terminal; t++) {
            eps = std::max((eps_min - eps_init) / (unsigned int)(num_of_frames * 0.10) * frame_count + eps_init, eps_min);

            if(frame_count % sync_interval == 0) sync();

            // --- //

            std::vector<double> state = sample_state(m, t);
            unsigned int action = eps_greedy_policy(state, eps);

            double diff = market->asset(MAIN_ASSET)->at(t+1) - market->asset(MAIN_ASSET)->at(t);
            if(diff > 0.00) diff = 1.00;
            else diff = -1.00;

            double observed_reward = diff * action_space[action];
            double expected_reward = observed_reward;
            if(t != terminal) {
                std::vector<double> next_state = sample_state(m, t+1);
                std::vector<double> target_q = target.predict(next_state);
                expected_reward += gamma * *std::max_element(target_q.begin(), target_q.end());

                std::vector<double>().swap(next_state);
                std::vector<double>().swap(target_q);
            }

            // --- //

            benchmark += diff;
            model += observed_reward;

            std::vector<double> agent_q = agent.predict(state);
            loss_sum += pow(expected_reward - agent_q[action], 2);
            mean_loss = loss_sum / (frame_count + 1);

            std::vector<double>().swap(agent_q);

            std::ofstream log("./res/log", std::ios::app);
            log << mean_loss << " " << benchmark << " " << model << " " << eps << " " << alpha << "\n";
            log.close();

            std::cout << "frame-" << frame_count << " @ " << market->ticker(MAIN_ASSET) << ": ";
            std::cout << "(eps=" << eps << ", alpha=" << alpha << ") ";
            std::cout << "action=" << action << " -> " << "observed=" << observed_reward << ", expected=" << expected_reward << ", ";
            std::cout << "benchmark=" << benchmark << ", model=" << model << "\n";

            memory.push_back(Memory(state, action, expected_reward));
            std::vector<double>().swap(state);

            frame_count++;

            // --- //

            if(memory.size() == memory_capacity) {
                std::vector<unsigned int> index(memory_capacity, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + batch_size, index.end());

                alpha = std::max((alpha_min - alpha_init) / (num_of_frames - memory_capacity) * (frame_count - memory_capacity) + alpha_init, alpha_min);

                for(unsigned int k: index)
                    sgd(memory[k], alpha, lambda);

                memory.erase(memory.begin(), memory.begin() + 1);

                std::vector<unsigned int>().swap(index);
            }
        }

        std::system("./python/graph.py");
    }

    std::vector<Memory>().swap(memory);
}

void Quant::sgd(Memory &memory, double alpha, double lambda) {
    std::vector<double> agent_q = agent.predict(*memory.state());
    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
        double partial_gradient = 0.00, gradient = 0.00;
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            if(l == agent.num_of_layers() - 1) {
                if(n == memory.action())
                    partial_gradient = -2.00 * (memory.expected_reward() - agent_q[n]);
                else
                    partial_gradient = -2.00 * (agent_q[n] - agent_q[n]);
            }
            else
                partial_gradient = agent.layer(l)->node(n)->err() * relu_prime(agent.layer(l)->node(n)->sum());

            double updated_bias = agent.layer(l)->node(n)->bias() - alpha * partial_gradient;
            agent.layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                if(l == 0)
                    gradient = partial_gradient * (*memory.state())[i];
                else {
                    gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                    agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                }

                gradient += 2.00 * lambda * agent.layer(l)->node(n)->weight(i);

                double updated_weight = agent.layer(l)->node(n)->weight(i) - alpha * gradient;
                agent.layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }

    std::vector<double>().swap(agent_q);
}
