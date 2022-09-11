#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <string>
#include <random>

#include "../lib/data.hpp"
#include "../lib/net.hpp"

#define MAIN_ASSET 0

void update_log(double mean_loss, double eps, double alpha, unsigned int frame, std::string ticker,
                unsigned int action, double observed_reward, double expected_reward, double benchmark, double model);

class Quant
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;

    std::vector<Market> *dataset;
    unsigned int num_of_frames;

    unsigned int look_back;
    std::vector<double> action_space;

    std::string checkpoint;

public:
    Quant() {}
    Quant(std::vector<Market> &_dataset, std::string _checkpoint): dataset(&_dataset), checkpoint(_checkpoint) {
        look_back = 100;
        action_space = std::vector<double>({-1.0, 0.0, 1.0});

        init({{500,450},{450,400},{400,350},{350,300},{300,250},{250,3}});
        load();

        num_of_frames = 0;
        for(unsigned int m = 0; m < dataset->size(); m++) {
            Market *market = &dataset->at(m);

            unsigned int start = look_back - 1;
            unsigned int terminal = market->asset(MAIN_ASSET)->size() - 2;
            num_of_frames += terminal - start + 1;
        }
    }
    ~Quant() {
        std::vector<double>().swap(action_space);
        save();
    }

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(Market *market, unsigned int t);

    unsigned int policy(std::vector<double> &state);
    unsigned int eps_greedy_policy(std::vector<double> &state, double eps);

    void build();
    void sgd(Memory &memory, double alpha, double lambda);

    void test();
    void run();

    void save();
    void load();
};

#endif