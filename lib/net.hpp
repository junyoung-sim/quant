#ifndef __NET_HPP_
#define __NET_HPP_

#include <cstdlib>
#include <vector>
#include <string>
#include <random>

double relu(double x);
double relu_prime(double x);

class Node
{
private:
    double b; // bias
    double s; // sum
    double z; // activation
    double e; // error
    std::vector<double> w;
public:
    Node() {}
    Node(unsigned int in) {
        init();
        b = 0.00;
        w = std::vector<double>(in, 0.00);
    }
    ~Node() {
        std::vector<double>().swap(w);
    }

    double bias();
    double sum();
    double act();
    double err();
    double weight(unsigned int index);

    void init();
    void set_bias(double val);
    void set_sum(double val);
    void set_act(double val);
    void add_err(double val);
    void set_weight(unsigned int index, double val);
};

// --- //

class Layer
{
private:
    std::vector<Node> n;
    unsigned int in;
    unsigned int out;
public:
    Layer() {}
    Layer(unsigned int _in, unsigned int _out): in(_in), out(_out) {
        n = std::vector<Node>(out, Node(in));
    }
    ~Layer() {
        std::vector<Node>().swap(n);
    }

    Node *node(unsigned int index);

    unsigned int in_features();
    unsigned int out_features();
};

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
    void init(std::default_random_engine &seed);

    unsigned int num_of_layers();
    Layer *layer(unsigned int index);

    std::vector<double> predict(std::vector<double> &x);

    void save(std::string path);
    void load(std::string path);
};

#endif
