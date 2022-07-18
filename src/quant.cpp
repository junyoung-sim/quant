
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "../lib/data.hpp"
#include "../lib/quant.hpp"

#define LONG 0
#define SHORT 1
#define OUT 2

void Quant::init(std::vector<std::vector<unsigned int>> shape) {
    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }

    agent.init(seed);
    sync();
}

void Quant::sync() {
    for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                double weight = agent.layer(l)->node(n)->weight(i);
                target.layer(l)->node(n)->set_weight(i, weight);
            }
        }
    }
}

bool Quant::sample_state(std::vector<double> &asset, std::vector<double> &vix, unsigned int t, unsigned int look_back, std::vector<double> &state) {
    std::vector<double> asset_price = {asset.begin(), asset.begin() + t + 1};
    std::vector<double> asset_macd = moving_average_convergence_divergence(asset_price, 12, 29);
    std::vector<double> asset_macd_signal = exponential_moving_average(asset_price, 9);
    std::vector<double> asset_sosc_ema = stochastic_oscillator(asset_price, 10, 10);

    std::vector<double> vix_price = {vix.begin(), vix.begin() + t + 1};
    std::vector<double> vix_sosc_ema = stochastic_oscillator(vix_price, 10, 10);

    range_normalize(asset_price);
    range_normalize(asset_macd);
    range_normalize(asset_macd_signal);
    range_normalize(asset_sosc_ema);
    range_normalize(vix_price);
    range_normalize(vix_sosc_ema);

    std::vector<double> asset_price_t = {asset_price.end() - look_back, asset_price.end()};
    std::vector<double> asset_macd_t = {asset_macd.end() - look_back, asset_macd.end()};
    std::vector<double> asset_macd_signal_t = {asset_macd_signal.end() - look_back, asset_macd_signal.end()};
    std::vector<double> asset_sosc_ema_t = {asset_sosc_ema.end() - look_back, asset_sosc_ema.end()};
    std::vector<double> vix_price_t = {vix_price.end() - look_back, vix_price.end()};
    std::vector<double> vix_sosc_ema_t = {vix_sosc_ema.end() - look_back, vix_sosc_ema.end()};

    std::vector<unsigned int> size = {static_cast<unsigned int>(asset_price_t.size()),
                                      static_cast<unsigned int>(asset_macd_t.size()),
                                      static_cast<unsigned int>(asset_macd_signal_t.size()),
                                      static_cast<unsigned int>(asset_sosc_ema_t.size()),
                                      static_cast<unsigned int>(vix_price_t.size()),
                                      static_cast<unsigned int>(vix_sosc_ema_t.size())};
    unsigned int min = *std::min_element(size.begin(), size.end());

    asset_price_t.erase(asset_price_t.begin(), asset_price_t.begin() + (asset_price_t.size() - min));
    asset_macd_t.erase(asset_macd_t.begin(), asset_macd_t.begin() + (asset_macd_t.size() - min));
    asset_macd_signal_t.erase(asset_macd_signal_t.begin(), asset_macd_signal_t.begin() + (asset_macd_signal_t.size() - min));
    asset_sosc_ema_t.erase(asset_sosc_ema_t.begin(), asset_sosc_ema_t.begin() + (asset_sosc_ema_t.size() - min));
    vix_price_t.erase(vix_price_t.begin(), vix_price_t.begin() + (vix_price_t.size() - min));
    vix_sosc_ema_t.erase(vix_sosc_ema_t.begin(), vix_sosc_ema_t.begin() + (vix_sosc_ema_t.size() - min));

    state.insert(state.end(), asset_price_t.begin(), asset_price_t.end());
    state.insert(state.end(), asset_macd_t.begin(), asset_macd_t.end());
    state.insert(state.end(), asset_macd_signal_t.begin(), asset_macd_signal_t.end());
    state.insert(state.end(), asset_sosc_ema_t.begin(), asset_sosc_ema_t.end());
    state.insert(state.end(), vix_price_t.begin(), vix_price_t.end());
    state.insert(state.end(), vix_sosc_ema_t.begin(), vix_sosc_ema_t.end());

    return t + 1 == asset.size() - 1;
}

std::vector<unsigned int> Quant::eps_greedy_policy(std::vector<double> &state, double eps) {
    unsigned int action, action_type;
    double explore = (double)rand() / RAND_MAX;

    if(explore < eps) {
        action = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
        action_type = 0;
    }
    else {
        std::vector<double> agent_q = agent.predict(state);
        action = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();
        action_type = 1;
    }

    return std::vector<unsigned int>({action, action_type});
}

void Quant::optimize(std::vector<double> &asset, std::vector<double> &vix, double eps_init, double eps_min, double alpha_init, double alpha_min,
                     double gamma, unsigned int memory_capacity, unsigned int batch_size, unsigned int sync_interval, unsigned int look_back, std::string checkpoint) {
    double eps = eps_init;
    double alpha = alpha_init;

    std::vector<Memory> replay_memory;

    unsigned int training_count = 0;
    double loss_sum = 0.00, mean_loss = 0.00;
    double benchmark = 1.00, model = 1.00;

    for(unsigned int t = look_back - 1; t <= asset.size() - 2; t++) {
        std::vector<double> state;
        bool terminal = sample_state(asset, vix, t, look_back, state);

        std::vector<unsigned int> act = eps_greedy_policy(state, eps);
        unsigned int action = act[0], action_type = act[1];

        double diff = (asset[t+1] - asset[t]) / asset[t];
        double observed_reward;
        if(action == LONG)
            observed_reward = diff;
        else if(action == SHORT)
            observed_reward = -diff;
        else
            observed_reward = 0.00;

        double expected_reward = observed_reward;
        if(!terminal) {
            std::vector<double> next_state;
            sample_state(asset, vix, t+1, look_back, next_state);

            std::vector<double> target_q = target.predict(next_state);
            expected_reward += gamma * *std::max_element(target_q.begin(), target_q.end());
        }

        if(training_count > 0) {
            std::vector<double> agent_q = agent.predict(state);
            loss_sum += pow(expected_reward - agent_q[action], 2);
            mean_loss = loss_sum / training_count;

            benchmark *= 1.00 + diff;
            model *= 1.00 + observed_reward;

            std::ofstream out("./data/log", std::ios::app);
            out << mean_loss << " " << benchmark << " " << model << " " << eps << " " << alpha << "\n";
            out.close();
        }

        std::cout << "@frame=" << t << ": (diff=" << diff << ") (action=" << action << ") ";
        std::cout << "(observed=" << observed_reward << ") (expected=" << expected_reward << ") ";
        std::cout << "(benchmark=" << benchmark << ") (model=" << model << ")\n";

        replay_memory.push_back(Memory(state, action, expected_reward));

        if(replay_memory.size() == memory_capacity) {            
            std::vector<unsigned int> index(memory_capacity, 0);
            std::iota(index.begin(), index.end(), 0);
            std::shuffle(index.begin(), index.end(), seed);
            index.erase(index.begin() + batch_size, index.end());

            for(unsigned int k: index) {
                std::vector<double> agent_q = agent.predict(*replay_memory[k].state());
                for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
                    double partial_gradient = 0.00, gradient = 0.00;
                    for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
                        if(l == agent.num_of_layers() - 1) {
                            if(n == replay_memory[k].action())
                                partial_gradient = -2.00 * (replay_memory[k].reward() - agent_q[n]);
                            else
                                partial_gradient = -2.00 * (agent_q[n] - agent_q[n]);
                        }
                        else
                            partial_gradient = agent.layer(l)->node(n)->err() * relu_prime(agent.layer(l)->node(n)->sum());

                        double updated_bias = agent.layer(l)->node(n)->bias() - alpha * partial_gradient;
                        agent.layer(l)->node(n)->set_bias(updated_bias);

                        for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                            if(l == 0)
                                gradient = partial_gradient * (*replay_memory[k].state())[i];
                            else {
                                gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                                agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                            }

                            double updated_weight = agent.layer(l)->node(n)->weight(i) - alpha * gradient;
                            agent.layer(l)->node(n)->set_weight(i, updated_weight);
                        }
                    }
                }
            }

            replay_memory.erase(replay_memory.begin(), replay_memory.begin() + 1);

            training_count++;
            eps = (eps_min - eps_init) / (asset.size() - look_back - 2 - memory_capacity) * training_count + eps_init;
            alpha = (alpha_min - alpha_init) / (asset.size() - look_back - 2 - memory_capacity) * training_count + alpha_init;

            if(training_count % sync_interval == 0) {
                sync();
                save(checkpoint);

                std::system("./python/graph.py");
            }
        }
    }

    save(checkpoint);
}

void Quant::save(std::string checkpoint) {
    agent.save(checkpoint);
}

