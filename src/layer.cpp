
#include <cstdlib>
#include <vector>
#include <string>

#include "../lib/layer.hpp"

Node *Layer::node(unsigned int index) {
    return &n[index];
}

unsigned int Layer::in_features() {
    return in;
}

unsigned int Layer::out_features() {
    return out;
}

