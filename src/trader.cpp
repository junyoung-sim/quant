
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>

#include "../lib/bar.hpp"
#include "../lib/trader.hpp"

void Trader::init(std::vector<std::vector<unsigned int>> shape) {
    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }

    seed.seed(std::chrono::system_clock::now().time_since_epoch().count());
    agent.initialize(seed);
    sync();
}

bool Trader::sample_state(std::vector<double> &series, unsigned int t, unsigned int look_back, std::vector<double> &state) {
    std::vector<double> price = {series.begin() + t, series.begin() + t + look_back};
    std::vector<double> macd = moving_average_convergence_divergence(price);
    std::vector<double> sosc = stochastic_oscillator(price, 14);
    std::vector<double> rsi = relative_strength_index(price, 14);

    std::vector<unsigned int> size = {static_cast<unsigned int>(price.size()),
                                      static_cast<unsigned int>(macd.size()),
                                      static_cast<unsigned int>(sosc.size()),
                                      static_cast<unsigned int>(rsi.size())};
    unsigned int min_size = *std::min_element(size.begin(), size.end());

    price.erase(price.begin(), price.begin() + (price.size() - min_size));
    macd.erase(macd.begin(), macd.begin() + (macd.size() - min_size));
    sosc.erase(sosc.begin(), sosc.begin() + (sosc.size() - min_size));
    rsi.erase(rsi.begin(), rsi.begin() + (rsi.size() - min_size));

    standardize(price);
    standardize(macd);

    state.insert(state.end(), price.begin(), price.end());
    state.insert(state.end(), macd.begin(), macd.end());
    state.insert(state.end(), sosc.begin(), sosc.end());
    state.insert(state.end(), rsi.begin(), rsi.end());

    return t + look_back == series.size() - 1; // terminal state
}

unsigned int Trader::epsilon_greedy_policy(std::vector<double> &state, double EPSILON) {
    unsigned int action;
    double explore = (double)rand() / RAND_MAX;
    // e-greedy policy
    if(explore < EPSILON)
        action = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
    else {
        std::vector<double> agent_q = agent.predict(state);
        action = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();
    }

    return action;
}

void Trader::sync() {
    for(unsigned int l = 0; l < agent.num_of_layers(); l++)
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++)
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++)
                target.layer(l)->node(n)->set_weight(i, agent.layer(l)->node(n)->weight(i));
}

void Trader::optimize(std::vector<double> &series) {
    // hyperparameters
    double EPSILON_INIT = 0.90;
    double EPSILON_MIN = 0.01;
    double EPSILON_DECAY = 0.99;
    double EPSILON = EPSILON_INIT;

    double GAMMA = 0.50;

    unsigned int MEMORY_CAPACITY = 10000;
    std::vector<std::vector<double>> state_memory;
    std::vector<unsigned int> action_memory;
    std::vector<double> reward_memory;
    std::vector<std::vector<double>> next_state_memory;

    unsigned int ITERATION = 10;
    unsigned int BATCH_SIZE = 64;
    double ALPHA_INIT = 0.0001;
    double ALPHA_MIN = 0.00001;
    double ALPHA_DECAY = 0.99;
    double ALPHA = ALPHA_INIT;

    unsigned int SYNC_INTERVAL = 5000;

    unsigned int LOOK_BACK = 60;
    init({{140,140}, {140,70}, {70,3}});

    // partition series (train, validation, test)
    std::vector<double> train_series = {series.begin(), series.begin() + (int)(series.size() * 0.98)};
    std::vector<double> test_series = {series.begin() + (int)(series.size() * 0.98), series.end()};;
    std::vector<double>().swap(series);

    for(unsigned int t = 0; t <= train_series.size() - LOOK_BACK - 1; t++) {
        progress_bar(t, train_series.size() - LOOK_BACK, "@frame=" + std::to_string(t));

        // sample state space
        std::vector<double> state;
        bool terminal = sample_state(train_series, t, LOOK_BACK, state);

        // compute action given state space
        double epsilon_decay_exp = log10(EPSILON_MIN / EPSILON_INIT) / log10(EPSILON_DECAY) * t / (train_series.size() - LOOK_BACK);
        EPSILON = EPSILON_INIT * pow(EPSILON_DECAY, epsilon_decay_exp);

        unsigned int action = epsilon_greedy_policy(state, EPSILON); // LONG (0), HOLD (1), SHORT (2)

        // observe reward
        double diff = (train_series[t+LOOK_BACK] - train_series[t+LOOK_BACK-1]) * 100 / train_series[t+LOOK_BACK-1];
        double reward;
        if(action == LONG)
            reward = diff;
        else if(action == SHORT)
            reward = -1.00 * diff;
        else
            reward = 0.00;

        // update replay memory
        state_memory.push_back(state);
        action_memory.push_back(action);
        reward_memory.push_back(reward);

        if(!terminal) {
            std::vector<double> next_state;
            sample_state(train_series, t+1, LOOK_BACK, next_state);
            next_state_memory.push_back(next_state);
        }

        // learn from replay memory
        if(state_memory.size() == MEMORY_CAPACITY) {
            // sample batch
            std::vector<unsigned int> index(MEMORY_CAPACITY, 0);
            std::iota(index.begin(), index.end(), 0);
            std::shuffle(index.begin(), index.end(), seed);
            index.erase(index.begin() + BATCH_SIZE, index.end());

            for(unsigned int itr = 1; itr <= ITERATION; itr++) {
                for(unsigned int k: index) {
                    // compute expected reward
                    double expected_reward = reward_memory[k];
                    if(k <= next_state_memory.size() - 1) { // non-terminal state
                        std::vector<double> target_q = target.predict(next_state_memory[k]);
                        expected_reward += GAMMA * *std::max_element(target_q.begin(), target_q.end());
                    }

                    // SGD
                    std::vector<double> agent_q = agent.predict(state_memory[k]);
                    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
                        unsigned int start = 0, end = agent.layer(l)->out_features();
                        if(l == agent.num_of_layers() - 1) {
                            start = action_memory[k];
                            end = start + 1;
                        }

                        double partial_gradient = 0.00, gradient = 0.00;
                        for(unsigned int n = start; n < end; n++) {
                            if(l == agent.num_of_layers() - 1)
                                partial_gradient = -2.00 * (expected_reward - agent_q[n]);
                            else
                                partial_gradient = agent.layer(l)->node(n)->err() * relu_prime(agent.layer(l)->node(n)->sum());

                            double updated_bias = agent.layer(l)->node(n)->bias() - ALPHA * partial_gradient;
                            agent.layer(l)->node(n)->set_bias(updated_bias);

                            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                                if(l == 0)
                                    gradient = partial_gradient * state_memory[k][i];
                                else {
                                    gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                                    agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                                }

                                double updated_weight = agent.layer(l)->node(n)->weight(i) - ALPHA * gradient;
                                agent.layer(l)->node(n)->set_weight(i, updated_weight);
                            }
                        }
                    }
                }
            }

            state_memory.erase(state_memory.begin(), state_memory.begin() + 1);
            action_memory.erase(action_memory.begin(), action_memory.begin() + 1);
            reward_memory.erase(reward_memory.begin(), reward_memory.begin() + 1);
            next_state_memory.erase(next_state_memory.begin(), next_state_memory.begin() + 1);
        }

        if(t % SYNC_INTERVAL == 0) sync();
    }
}

