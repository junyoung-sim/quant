#ifndef __DATA_HPP_
#define __DATA_HPP_

#include <cstdlib>
#include <vector>
#include <string>

std::vector<double> read_csv(std::string path, std::string column);

// --- //

void standardize(std::vector<double> &series);

std::vector<double> exponential_moving_average(std::vector<double> &series, unsigned int period);
std::vector<double> moving_average_convergence_divergence(std::vector<double> &series, unsigned int fast_period, unsigned int slow_period);

// --- //

class Market
{
private:
    std::vector<std::string> tickers;
    std::vector<std::vector<double>> assets;
public:
    Market() {}
    Market(std::vector<std::string> _tickers) {
        tickers.swap(_tickers);
        std::string cmd = "./python/clean.py ";
        for(unsigned int i = 0; i < tickers.size(); i++) {
            cmd += tickers[i];
            if(i != tickers.size() - 1)
                cmd += " ";
        }
        std::system(cmd.c_str());

        for(std::string &ticker: tickers) {
            std::vector<double> asset = read_csv("./data/cleaned.csv", ticker);
            assets.push_back(asset);

            std::vector<double>().swap(asset);
        }
    }
    ~Market() {
        std::vector<std::string>().swap(tickers);
        std::vector<std::vector<double>>().swap(assets);
    }

    unsigned int num_of_assets();

    std::string ticker(unsigned int i);
    std::vector<double> *asset(unsigned int i);
};

// --- //

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

#endif