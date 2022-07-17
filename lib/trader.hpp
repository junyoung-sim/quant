#ifndef __DQN_HPP_
#define __DQN_HPP_

#define LONG 0
#define SHORT 1
#define OUT 2

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
        init({{50,50},{50,25},{25,3}});
    }
    ~Trader() {}

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    bool sample_state(std::vector<double> &asset, std::vector<double> &vix, unsigned int t, unsigned int look_back, std::vector<double> &state);
    std::tuple<unsigned int, double> epsilon_greedy_policy(std::vector<double> &state, double EPSILON);
    void optimize(std::vector<double> &asset, std::vector<double> &vix, double EPSILON_INIT, double EPSILON_MIN, double ALPHA_INIT, double ALPHA_MIN,
                  double GAMMA, unsigned int MEMORY_CAPACITY, unsigned int BATCH_SIZE, unsigned int SYNC_INTERVAL, unsigned int LOOK_BACK, std::string checkpoint);

    void save(std::string path);
    void load(std::string path);
};

#endif
