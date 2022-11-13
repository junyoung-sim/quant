#ifndef __DATA_HPP_
#define __DATA_HPP_

#include <cstdlib>
#include <vector>
#include <string>

std::vector<std::vector<double>> read_csv(std::string path);

void standardize(std::vector<double> &dat);

class Market
{
private:
    std::vector<std::string> tickers;
    std::vector<std::vector<double>> assets;
public:
    Market() {}
    Market(std::vector<std::string> _tickers, std::string path) {
        tickers.swap(_tickers);
        assets = read_csv(path);
    }
    ~Market() {
        std::vector<std::string>().swap(tickers);
        std::vector<std::vector<double>>().swap(assets);
    }

    std::string ticker(unsigned int i);

    unsigned int num_of_assets();
    std::vector<double> *asset(unsigned int i);
};

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

    std::vector<double> *state();
    unsigned int action();
    double expected_reward();
};

#endif