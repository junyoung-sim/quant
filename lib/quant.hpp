#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <random>
#include <string>

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

    unsigned int look_back;
    unsigned int decision_interval;

    std::string checkpoint;

public:
    Quant(Market &_market, std::string _checkpoint): checkpoint(_checkpoint) {
        market = &_market;
        look_back = 50;
        decision_interval = 10;
        init({{250,250},{250,250},{250,250},{250,150},{150,3}});

        load(checkpoint);
    }
    ~Quant() {
        save(checkpoint);
    }

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(unsigned int t);

    unsigned int policy(std::vector<double> &state);
    unsigned int eps_greedy_policy(std::vector<double> &state, double eps);

    void optimize(double eps_init, double eps_min, double alpha_init, double alpha_min,
                  double gamma, unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval);
    void sgd(Memory &memory, double alpha);

    void save(std::string path);
    void load(std::string path);
};

#endif