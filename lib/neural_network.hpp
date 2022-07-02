#ifndef __NEURAL_NETWORK_HPP_
#define __NEURAL_NETWORK_HPP_

#include <vector>
#include <string>
#include <random>

#include "layer.hpp"

double relu(double x);
double relu_prime(double x);

double mse(std::vector<double> &y, std::vector<double> &yhat);

// --- //

class NeuralNetwork
{
private:
    std::vector<Layer> layers;
public:
    NeuralNetwork() {}
    ~NeuralNetwork() {
        std::vector<Layer>().swap(layers);
    }

    void add_layer(unsigned int in, unsigned int out);
    void initialize(std::default_random_engine &seed);

    unsigned int num_of_layers();
    Layer *layer(unsigned int index);

    std::vector<double> predict(std::vector<double> &x);
};

#endif
