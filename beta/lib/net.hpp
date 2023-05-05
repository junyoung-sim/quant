#ifndef __NET_HPP_
#define __NET_HPP_

#include <vector>
#include <string>
#include <random>

double relu(double x);
double relu_prime(double x);

class Node
{
private:
    double b;
    double s;
    double z;
    double e;
    std::vector<double> w;
public:
    Node() {}
    Node(unsigned int in) {
        init();
        b = 0.00;
        w = std::vector<double>(in, 0.00);
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

    unsigned int in_features();
    unsigned int out_features();

    Node *node(unsigned int index);
};

class NeuralNetwork
{
private:
    std::vector<Layer> layers;
public:
    NeuralNetwork() {}

    void add_layer(unsigned int in, unsigned int out);
    void init(std::default_random_engine &seed);

    unsigned int num_of_layers();
    Layer *layer(unsigned int index);

    std::vector<double> predict(std::vector<double> &x);
};

#endif