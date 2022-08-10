#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <string>
#include <random>

#include "../lib/data.hpp"
#include "../lib/net.hpp"

class Quant
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;

    std::vector<Market> *market_dataset;
    unsigned int look_back;
    std::vector<double> action_space;

    std::string checkpoint;

public:
    Quant() {}
    Quant(std::vector<Market> &_market_dataset, std::string _checkpoint): checkpoint(_checkpoint) {
        market_dataset = &_market_dataset;
        look_back = 20;
        action_space = std::vector<double>({1.0, 0.0, -1.0}); // long, idle, short

        init({{100,100},{100,100},{100,100},{100,100},{100,50},{50,3}});
        load();
    }
    ~Quant() {
        std::vector<Market>().swap(*market_dataset);
        std::vector<double>().swap(action_space);
    }

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(unsigned int market_id, unsigned int t);

    unsigned int policy(std::vector<double> &state);
    unsigned int eps_greedy_policy(std::vector<double> &state, double eps);

    void optimize(double eps_init, double eps_min, double alpha_init, double alpha_min,
                  double gamma, unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval);

    void save();
    void load();
};

#endif