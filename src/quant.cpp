
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>

#include "../lib/quant.hpp"

#define LONG 0
#define SHORT 1

void Quant::init(std::vector<std::vector<unsigned int>> shape, std::default_random_engine &seed) {
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

unsigned int Quant::eps_greedy_policy(std::vector<double> &state, double eps) {
    unsigned int action;
    double explore = (double)rand() / RAND_MAX;

    if(explore < eps)
        action = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
    else {
        std::vector<double> agent_q = agent.predict(state);
        action = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();

        std::vector<double>().swap(agent_q);
    }

    return action;
}

void Quant::optimize(std::vector<double> &series, double eps_init, double eps_min, double alpha_init, double alpha_min, double gamma,
                     unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval, std::string checkpoint, std::default_random_engine &seed) {
    double eps = 1.00;
    double alpha = alpha_init;

    std::vector<Memory> memory;

    unsigned int training_count = 0;
    double loss_sum = 0.00, mean_loss = 0.00;
    double model = 1.00;

    unsigned int start = 9;
    for(unsigned int t = start; t <= series.size() - 2; t++) {
        std::vector<double> state = sample_state(series, t);
        unsigned int action = eps_greedy_policy(state, eps);

        double diff = (series[t+1] - series[t]) / series[t];
        double observed_reward;
        if(action == LONG)
            observed_reward = diff;
        else
            observed_reward = -diff;

        double expected_reward = observed_reward;
        if(t != series.size() - 2) {
            std::vector<double> next_state = sample_state(series, t+1);
            std::vector<double> target_q = target.predict(next_state);
            expected_reward += gamma * *std::max_element(target_q.begin(), target_q.end());

            std::vector<double>().swap(next_state);
            std::vector<double>().swap(target_q);
        }

        if(training_count > 0) {
            std::vector<double> agent_q = agent.predict(state);
            loss_sum += pow(expected_reward - agent_q[action], 2);
            mean_loss = loss_sum / training_count;
            model *= 1.00 + observed_reward;

            std::ofstream log("./res/log", std::ios::app);
            log << mean_loss << " " << model << "\n";
            log.close();

            std::vector<double>().swap(agent_q);
        }

        memory.push_back(Memory(state, action, expected_reward));
        std::vector<double>().swap(state);

        std::cout << "@frame" << t << ": action = " << action << " -> observed = " << observed_reward << ", "
                  << "expected = " << expected_reward << ", model return = " << model << "\n";

        if(memory.size() == memory_capacity) {
            std::vector<unsigned int> index(memory_capacity, 0);
            std::iota(index.begin(), index.end(), 0);
            std::shuffle(index.begin(), index.end(), seed);
            index.erase(index.begin() + batch_size, index.end());

            for(unsigned int k: index)
                sgd(memory[k], alpha);

            memory.erase(memory.begin(), memory.begin() + 1);
            std::vector<unsigned int>().swap(index);

            training_count++;
            eps = (eps_min - eps_init) / (series.size() - start - 1 - memory_capacity) * training_count + eps_init;
            alpha = (alpha_min - alpha_init) / (series.size() - start - 1 - memory_capacity) * training_count + alpha_init;

            if(training_count % sync_interval == 0) sync();
        }
    }
}

void Quant::sgd(Memory &memory, double alpha) {
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

                double updated_weight = agent.layer(l)->node(n)->weight(i) - alpha * gradient;
                agent.layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }

    std::vector<double>().swap(agent_q);
}
