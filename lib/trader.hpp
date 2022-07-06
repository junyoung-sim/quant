#ifndef __DQN_HPP_
#define __DQN_HPP_

#define LONG 0
#define HOLD 1
#define SHORT 2

#include <vector>
#include <random>
#include <chrono>

#include "data.hpp"
#include "neural_network.hpp"

class Trader
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;
public:
    Trader() {}
    ~Trader() {}

    void init(std::vector<std::vector<unsigned int>> shape);

    bool sample_state(std::vector<double> &series, unsigned int t, unsigned int look_back, std::vector<double> &state);
    unsigned int epsilon_greedy_policy(std::vector<double> &state, double EPSILON);
    void sync();

    void optimize(std::vector<double> &series);
};

#endif
