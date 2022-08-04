#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <random>

#include "data.hpp"
#include "net.hpp"

#define LONG 0
#define SHORT 1
#define IDLE 2

#define MAIN_ASSET 0

class Quant
{
private:
    Market *market;
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;

    std::vector<double> kernel;

    unsigned int moving_average_period;
    unsigned int look_back;

public:
    Quant(Market &_market) {
        market = &_market;
        moving_average_period = 10;
        look_back = 25;

        init({{25,25},{25,20},{20,15},{15,3}});
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