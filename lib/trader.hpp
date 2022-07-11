#ifndef __DQN_HPP_
#define __DQN_HPP_

#define LONG 0
#define NONE 1
#define SHORT 2

#include <cstdlib>
#include <vector>
#include <random>
#include <tuple>

#include "data.hpp"
#include "neural_network.hpp"

class Trader
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;
public:
    Trader() {
        init({{125,100}, {100,50}, {50,3}});
    }
    ~Trader() {}

    void init(std::vector<std::vector<unsigned int>> shape);

    bool sample_state(std::vector<double> &series, unsigned int t, unsigned int look_back, std::vector<double> &state);
    std::tuple<unsigned int, double> epsilon_greedy_policy(std::vector<double> &state, double EPSILON);
    void sync();

    void optimize(std::vector<double> &series, std::string checkpoint);

    void save(std::string path);
    void load(std::string path);
};

#endif
