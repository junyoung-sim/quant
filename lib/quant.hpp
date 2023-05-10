#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <vector>
#include <string>
#include <random>
#include <map>

#include "../lib/data.hpp"
#include "../lib/net.hpp"

#define TICKER 0

typedef std::map<std::string, std::vector<std::vector<double>>> Environment; // (ticker, historical data)

struct Memory {
    std::vector<double> state;
    unsigned int action;
    double optimal;

    Memory(std::vector<double> &s, unsigned int a, double opt) {
        state.swap(s);
        action = a;
        optimal = opt;
    }
};

class Quant
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;

    unsigned int look_back; // number of market days observed in state
    unsigned int paa_window; // piecewise aggregate approximation window

    std::vector<double> action_space;

    std::string checkpoint;

public:
    Quant(std::string path): checkpoint(path) {
        look_back = 100; // 100 market days
        paa_window = 5; // 5 market days (weekly discretization)

        action_space = std::vector<double>({-1.0, 0.0, 1.0}); // short, idle, long

        init({{500,450},{450,400},{400,350},{350,300},{300,250},{250,3}});
        load();
    }
    
    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(std::vector<std::vector<double>> &dat, unsigned int t);

    unsigned int greedy(std::vector<double> &state);
    unsigned int epsilon_greedy(std::vector<double> &state, double eps);

    void build(std::vector<std::string> &tickers, Environment &env);
    void sgd(Memory &memory, double alpha, double lambda);

    void test(std::vector<std::string> &tickers, Environment &env);
    void run(std::vector<std::string> &tickers, Environment &env);

    void save();
    void load();
};

#endif