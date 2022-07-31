#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <random>
#include <string>

#include "../lib/data.hpp"
#include "../lib/net.hpp"

class Memory
{
private:
    std::vector<double> s;
    unsigned int a;
    double r;
public:
    Memory() {}
    Memory(std::vector<double> &state, unsigned int action, double expected_reward) {
        s.swap(state);
        a = action;
        r = expected_reward;
    }
    ~Memory() {
        std::vector<double>().swap(s);
    }

    std::vector<double> *state() { return &s; }
    unsigned int action() { return a; }
    double expected_reward() { return r; }
};

class Quant
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
public:
    Quant() {}
    ~Quant() {}

    void init(std::vector<std::vector<unsigned int>> shape, std::default_random_engine &seed);
    void sync();

    unsigned int eps_greedy_policy(std::vector<double> &state, double eps);
    void optimize(std::vector<double> &series, double eps_init, double eps_min, double alpha_init, double alpha_min, double gamma, unsigned int memory_capacity,
                  unsigned int batch_size, unsigned int sync_interval, unsigned int look_back, std::string checkpoint, std::default_random_engine &seed);
    void sgd(Memory &memory, double alpha);
};

#endif
