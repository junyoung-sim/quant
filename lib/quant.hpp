#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <string>
#include <random>

#include "neural_network.hpp"

class Memory
{
private:
    std::vector<double> s;
    unsigned int a;
    double r;
public:
    Memory(std::vector<double> &state, unsigned int action, double reward) {
        s.swap(state);
        a = action;
        r = reward;
    }
    ~Memory() {
        std::vector<double>().swap(s);
    }

    std::vector<double> *state() { return &s; }
    unsigned int action() { return a; }
    double reward() { return r; }
};

class Quant
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;
public:
    Quant() {
        init({{10,10},{10,5},{5,3}});
    }
    ~Quant() {}

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    bool sample_state(std::vector<double> &asset, unsigned int t, std::vector<double> &state);
    unsigned int eps_greedy_policy(std::vector<double> &state, double eps);

    void optimize(std::vector<double> &asset, double eps_init, double eps_min, double alpha_init, double alpha_min,
                  double gamma, unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval, std::string checkpoint);

    void save(std::string checkpoint);
    void load(std::string checkpoint);
};

#endif
