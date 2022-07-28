#ifndef __LAYER_HPP_
#define __LAYER_HPP_

#include <vector>
#include <string>

#include "node.hpp"

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

#endif
