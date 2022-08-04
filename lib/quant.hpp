#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <random>

#include "data.hpp"
#include "net.hpp"

#define LONG 0
#define SHORT 1

#define MAIN_ASSET 0

class Quant
{
private:
    Market *market;
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;

    unsigned int look_back;

public:
    Quant(Market &_market) {
        market = &_market;

        look_back = 50;
        init({{150,150},{150,100},{100,100},{100,50},{50,50},{50,25},{25,2}});
    }
    ~Quant() {}

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(unsigned int t);

    unsigned int policy(std::vector<double> &state);
    unsigned int eps_greedy_policy(std::vector<double> &state, double eps);

    void optimize(double eps_init, double eps_min, double alpha_init, double alpha_min, 
                  double gamma, unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval);
    void sgd(Memory &memory, double alpha);
};

#endif