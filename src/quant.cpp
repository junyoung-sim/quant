#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>
#include <map>

#include "../lib/quant.hpp"

void Quant::init(std::vector<std::vector<unsigned int>> shape) {
    // agent and target networks have idential shapes
    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }

    srand(time(NULL));
    seed.seed(std::chrono::system_clock::now().time_since_epoch().count());

    agent.init(seed); // He-initialization
    sync();
}

void Quant::sync() {
    // synchronize target network to agent network
    for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                double weight = agent.layer(l)->node(n)->weight(i);
                target.layer(l)->node(n)->set_weight(i, weight);
            }

            double bias = agent.layer(l)->node(n)->bias();
            target.layer(l)->node(n)->set_bias(bias);
        }
    }
}

std::vector<double> Quant::sample_state(std::vector<std::vector<double>> &dat, unsigned int t) {
    std::vector<double> state;
    for(int i = 0; i < dat.size(); i++) {
        std::vector<double> series = {dat[i].begin() + t + 1 - look_back, dat[i].begin() + t + 1};
        // discretize and standardize price series of each asset in dat
        piecewise_aggregate_approximation(series, paa_window);
        standardize(series);

        state.insert(state.end(), series.begin(), series.end());
    }
    return state;
}

unsigned int Quant::greedy(std::vector<double> &state) {
    std::vector<double> q = agent.predict(state);
    return std::max_element(q.begin(), q.end()) - q.begin(); // select action with highest predicted reward
}

unsigned int Quant::epsilon_greedy(std::vector<double> &state, double eps) {
    unsigned int action = greedy(state);
    double explore = (double)rand() / RAND_MAX;
    if(explore < eps) {
        action = rand() % action_space.size(); // explore a random action
        std::cout << "(E) ";
    }
    else
        std::cout << "(P) ";
    return action;
}

void Quant::build(std::vector<std::string> &tickers, Environment &env) {
    // compute size of the environment for all tickers
    unsigned int env_size = 0;
    for(std::string &ticker: tickers)
        env_size += env[ticker][TICKER].size();
    
    // learning hyperparamters
    
    double eps_init = 1.00; // initial exploration rate
    double eps_min = 0.10; // final exploration rate
    double gamma = 0.80; // long-term reward discount factor

    std::vector<Memory> replay_memory; // store past experiences
    unsigned int capacity = (unsigned int)(env_size * 0.10); // replay memory capacity
    unsigned int batch_size = 10; // batch size for SGD on agent network

    double alpha_init = 0.00001; // initial learning rate for SGD
    double alpha_min = 0.00000001; // final learning rate for SGD
    double alpha_decay = log(alpha_min) - log(alpha_init); // learning rate exponential decay rate
    double lambda = 0.10; // L2 regularization for SGD

    double eps = eps_init; // exploration rate
    double alpha = alpha_init; // learning for SGD

    unsigned int experiences = 0; // total number of states observed during training
    double rss = 0.00, mse = 0.00; // residual squared sum, mean squared error

    // train deep q-network trading model

    std::shuffle(tickers.begin(), tickers.end(), seed); // randomize order of training

    for(std::string &ticker: tickers) {
        unsigned int start = look_back - 1; // starting index
        unsigned int terminal = env[ticker][TICKER].size() - 2; // terminal index

        std::ofstream out("./res/log");
        double benchmark = 1.00, model = 1.00;

        for(unsigned int t = start; t <= terminal; t++) {
            // exploration rate linear decay (reach minimum when replay memory fills up for the first time)
            if(experiences <= capacity)
                eps = (eps_min - eps_init) / capacity * experiences + eps_init;
            
            std::vector<double> state = sample_state(env[ticker], t); // sample current state
            unsigned int action = epsilon_greedy(state, eps); // select action
            double q = agent.layer(agent.num_of_layers() - 1)->node(action)->sum(); // predicted q-value

            // observe discrete reward from daily p&l
            double diff = (env[ticker][TICKER][t+1] - env[ticker][TICKER][t]) / env[ticker][TICKER][t];
            double observed_reward = (diff >= 0.00 ? action_space[action] : -action_space[action]);

            // estimate optimal q-value
            std::vector<double> next_state = sample_state(env[ticker], t+1); // sample next state
            std::vector<double> tq = target.predict(next_state); // estimate long-term reward
            double optimal = observed_reward + gamma * *std::max_element(tq.begin(), tq.end());

            // update historical return-on-investment
            benchmark *= 1.00 + diff;
            model *= 1.00 + diff * action_space[action];
            out << benchmark << " " << model << " " << action << "\n";

            rss += pow(optimal - q, 2); // update model cost
            mse = rss / ++experiences;

            std::cout << std::fixed;
            std::cout.precision(15);
            std::cout << "(LOSS=" << mse << ", EPS=" << eps << ", ALPHA=" << alpha << ") ";
            std::cout << "T=" << t << " @ " << ticker << " ACTION=" << action << " ";
            std::cout << "-> OBS=" << observed_reward << " OPT=" << optimal << " ";
            std::cout << "BASE=" << benchmark << " " << "MOD=" << model << "\n";

            Memory memory(state, action, optimal); // save experience
            replay_memory.push_back(memory);

            if(replay_memory.size() == capacity) {
                std::vector<unsigned int> index(capacity, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + batch_size, index.end()); // randomly select 10 experiences

                // learning rate exponential decay
                alpha = alpha_init * exp(alpha_decay * (experiences - capacity) / (env_size - capacity));

                for(unsigned int k: index)
                    sgd(replay_memory[k], alpha, lambda); // update agent network
                
                replay_memory.erase(replay_memory.begin());
            }
        } sync(); // synchronize after each ticker for stability

        out.close();
        std::system(("./python/log.py " + ticker).c_str()); // output historical return-on-investment
    
    } save();
}

void Quant::sgd(Memory &memory, double alpha, double lambda) { // stochastic gradient descent (mse)
    std::vector<double> q = agent.predict(memory.state);
    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
        double partial_gradient = 0.00, gradient = 0.00;
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            if(l == agent.num_of_layers() - 1 && n != memory.action) continue; // ignore non-selected actions
            else {
                if(l == agent.num_of_layers() - 1)
                    partial_gradient = -2.00 * (memory.optimal - q[n]);
                else
                    partial_gradient = agent.layer(l)->node(n)->err() * relu_prime(agent.layer(l)->node(n)->sum());

                double updated_bias = agent.layer(l)->node(n)->bias() - alpha * partial_gradient;
                agent.layer(l)->node(n)->set_bias(updated_bias);

                for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                    if(l == 0)
                        gradient = partial_gradient * memory.state[i];
                    else {
                        gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                        agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                    }

                    gradient += lambda * agent.layer(l)->node(n)->weight(i);

                    double updated_weight = agent.layer(l)->node(n)->weight(i) - alpha * gradient;
                    agent.layer(l)->node(n)->set_weight(i, updated_weight);
                }
            }
        }
    }
}

void Quant::test(std::vector<std::string> &tickers, Environment &env) {
    for(std::string &ticker: tickers) {
        unsigned int start = look_back - 1;
        unsigned int terminal = env[ticker][TICKER].size() - 2;

        // track benchmark and model's historical return-on-investment
        std::ofstream out("./res/log");
        double benchmark = 1.00, model = 1.00;

        for(unsigned int t = start; t <= terminal; t++) {
            std::vector<double> state = sample_state(env[ticker], t);
            unsigned int action = greedy(state); // test the model's greedy policy

            // observe daily p&l of benchmark and model
            double diff = (env[ticker][TICKER][t+1] - env[ticker][TICKER][t]) / env[ticker][TICKER][t];
            benchmark *= 1.00 + diff;
            model *= 1.00 + diff * action_space[action];
            out << benchmark << " " << model << " " << action << "\n";

            // display progress
            std::cout << std::fixed;
            std::cout.precision(15);
            std::cout << "T=" << t << " @ " << ticker << " ACTION=" << action << " ";
            std::cout << "DIFF=" << diff << " BASE=" << benchmark << " MOD=" << model << "\n";
        }

        out.close();
        std::system(("./python/log.py " + ticker).c_str()); // output historical return-on-investment
        std::system(("./python/stats.py push " + ticker).c_str()); // analyze historical performance
    }

    std::system("./python/stats.py summary"); // summarize historical performance
}

void Quant::run(std::vector<std::string> &tickers, Environment &env) {
    double model_daily = 0.00;
    std::vector<int> action_count = {0, 0, 0};
    for(std::string &ticker: tickers) {
        unsigned int start = env[ticker][TICKER].size() - 252;
        unsigned int terminal = env[ticker][TICKER].size() - 1;

        std::ofstream out("./res/log");
        double benchmark = 1.00, model = 1.00;

        for(unsigned int t = start; t <= terminal; t++) {
            std::vector<double> state = sample_state(env[ticker], t);
            unsigned int action = greedy(state);

            if(t != terminal) {
                double diff = (env[ticker][TICKER][t+1] - env[ticker][TICKER][t]) / env[ticker][TICKER][t];
                benchmark *= 1.00 + diff;
                model *= 1.00 + diff * action_space[action];
                out << benchmark << " " << model << " " << action << "\n";

                if(t == terminal - 1)
                    model_daily += diff * action_space[action];
            }
            else {
                std::ofstream sout("./res/state");
                for(double &x: state)
                    sout << x << "\n";
                sout << action;
                sout.close();

                action_count[action]++;
            }
        }

        out.close();
        std::system(("./python/run.py " + ticker).c_str()); // output recent performance
    }

    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "    MODEL DAILY P&L = " << model_daily / tickers.size() * 100 << "%\n\n";
    std::cout << "Action (0) - Short: " << (double)action_count[0] * 100 / tickers.size() << "%\n";
    std::cout << "Action (1) - Idle : " << (double)action_count[1] * 100 / tickers.size() << "%\n";
    std::cout << "Action (2) - Long : " << (double)action_count[2] * 100 / tickers.size() << "%\n";
}

void Quant::save() {
    std::ofstream out(checkpoint);
    if(out.is_open()) {
        for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
            for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
                for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++)
                    out << agent.layer(l)->node(n)->weight(i) << " ";
                out << agent.layer(l)->node(n)->bias() << "\n";
            }
        }

        out.close();
    }
}

void Quant::load() {
    std::ifstream out(checkpoint);
    if(out.is_open()) {
        for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
            for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
                std::string line;
                std::getline(out, line);

                for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                    double weight = std::stod(line.substr(0, line.find(" ")));
                    agent.layer(l)->node(n)->set_weight(i, weight);

                    line = line.substr(line.find(" ") + 1);
                }

                double bias = std::stod(line);
                agent.layer(l)->node(n)->set_bias(bias);
            }
        }

        sync();
        out.close();
    }
}