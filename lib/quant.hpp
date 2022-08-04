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
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;

    unsigned int moving_average_period;
    unsigned int look_back;

public:
    Quant() {
        moving_average_period = 10;
        look_back = 25;

        init({{100,80},{80,40},{40,20},{20,2}});
    }
    ~Quant() {}

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(Market &market, unsigned int t);
    //unsigned int eps_greedy_policy(std::vector<double> &state, double eps);

    //void optimize(Market &market, double eps_init, double eps_min, double alpha_init, double alpha_min,
    //              double gamma, unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval);
};

#endif