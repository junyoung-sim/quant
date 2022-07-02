
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>

#include "../lib/bar.hpp"
#include "../lib/dqn.hpp"

void DQN::sync() {
    for(unsigned int l = 0; l < agent.num_of_layers(); l++)
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++)
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++)
                target.layer(l)->node(n)->set_weight(i, agent.layer(l)->node(n)->weight(i));
}

unsigned int DQN::epsilon_greedy_policy(std::vector<double> &state, double EPSILON) {
    unsigned int action;
    double explore = (double)rand() / RAND_MAX;
    // e-greedy policy
    if(explore < EPSILON)
        action = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
    else {
        std::vector<double> agent_q = agent.predict(state);
        action = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();

        std::vector<double>().swap(agent_q);
    }

    return action;
}

void DQN::optimize(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward) {
    double EPSILON_INIT = 0.90;
    double EPSILON_MIN = 0.01;
    double EPSILON_DECAY = 0.99;
    double EPSILON = EPSILON_INIT;

    double GAMMA = 0.30;

    unsigned int BATCH_SIZE = 512;
    std::vector<unsigned int> action_memory;

    double ALPHA_INIT = 0.0001;
    double ALPHA_MIN = 0.00001;
    double ALPHA_DECAY = 0.99;
    double ALPHA = ALPHA_INIT;

    unsigned int EPOCH = 10;

    unsigned int SYNC_INTERVAL = 2000;

    for(unsigned int frame = 0; frame < state.size(); frame++) {
        double epsilon_decay_exp = log10(EPSILON_MIN / EPSILON_INIT) / EPSILON_DECAY * frame / state.size();
        EPSILON *= pow(EPSILON_DECAY, epsilon_decay_exp);

        unsigned int action = epsilon_greedy_policy(state[frame], EPSILON); // e-greedy policy
        action_memory.push_back(action);

        // batch learning
        if(action_memory.size() >= BATCH_SIZE) {
            std::vector<unsigned int> batch(action_memory.size(), 0);
            std::iota(batch.begin(), batch.end(), 0);
            std::shuffle(batch.begin(), batch.end(), seed);

            double alpha_decay_exp = log10(ALPHA_MIN / ALPHA_INIT) / ALPHA_DECAY * (frame - BATCH_SIZE) / (state.size() - BATCH_SIZE);
            ALPHA *= pow(ALPHA_DECAY, alpha_decay_exp);

            for(unsigned int epoch = 1; epoch <= EPOCH; epoch++) {
                for(unsigned int index = 0; index < BATCH_SIZE; index++) {
                    unsigned int k = batch[index];
                    unsigned int action = action_memory[k];
                    // compute expected reward (finite bellman equation)
                    double expected_reward = reward[k][action];
                    if(k != state.size() - 1) {
                        std::vector<double> target_q = target.predict(state[k+1]);
                        expected_reward += GAMMA * *std::max_element(target_q.begin(), target_q.end());

                        std::vector<double>().swap(target_q);
                    }

                    // SGD
                    std::vector<double> agent_q = agent.predict(state[k]);
                    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
                        unsigned int start = 0, end = agent.layer(l)->out_features();
                        if(l == agent.num_of_layers() - 1) {
                            start = action;
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
                                    gradient = partial_gradient * state[k][i];
                                else {
                                    gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                                    agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                                }

                                double updated_weight = agent.layer(l)->node(n)->weight(i) - ALPHA * gradient;
                                agent.layer(l)->node(n)->set_weight(i, updated_weight);
                            }
                        }
                    }

                    std::vector<double>().swap(agent_q);
                }
            }

            std::vector<unsigned int>().swap(batch);

            std::vector<double> performance = evaluate_agent(state, reward, GAMMA);
            double mean_loss = performance[0], mean_reward = performance[1];

            progress_bar(frame, state.size(), "(frame=" + std::to_string(frame) + ") L = " + std::to_string(mean_loss) + ", R = " + std::to_string(mean_reward));

            std::ofstream out("./data/training_performance", std::ios::app);
            out << mean_loss << " " << mean_reward << "\n";
            out.close();

            std::vector<double>().swap(performance);
        }

        if(frame % SYNC_INTERVAL == 0) sync();
    }

    std::vector<unsigned int>().swap(action_memory);
}

std::vector<double> DQN::evaluate_agent(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward, double GAMMA) {
    double mean_loss = 0.00, mean_reward = 0.00;
    for(unsigned int frame = 0; frame < state.size(); frame++) {
        std::vector<double> agent_q = agent.predict(state[frame]);
        unsigned int action = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();

        double expected_reward = reward[frame][action];
        if(frame != state.size() - 1) {
            std::vector<double> target_q = target.predict(state[frame+1]);
            expected_reward += GAMMA * *std::max_element(target_q.begin(), target_q.end());

            std::vector<double>().swap(target_q);
        }

        mean_loss += pow(expected_reward - agent_q[action], 2);
        mean_reward += reward[frame][action];

        std::vector<double>().swap(agent_q);
    }

    mean_loss /= state.size();
    mean_reward /= state.size();

    return std::vector<double>({mean_loss, mean_reward});
}

