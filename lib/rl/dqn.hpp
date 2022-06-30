#ifndef __DQN_HPP_
#define __DQN_HPP_

#include <vector>
#include <random>
#include <chrono>

#include "neural_network.hpp"

class DQN
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;
public:
    DQN() {}
    DQN(std::vector<std::vector<unsigned int>> shape) {
        for(unsigned int l = 0; l < shape.size(); l++) {
            unsigned int in = shape[l][0], out = shape[l][1];
            agent.add_layer(in, out);
            target.add_layer(in, out);
        }

        seed.seed(std::chrono::system_clock::now().time_since_epoch().count());
        agent.initialize(seed);
        sync();
    }

    void sync();
    unsigned int select_action(std::vector<double> &state, double EPSILON);

    std::vector<double> evaluate_agent(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward, double GAMMA);
    void optimize(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward);
};

#endif
