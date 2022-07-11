
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

#include "../lib/neural_network.hpp"

/* assume implementation-specific architecture is hard-coded and initialized */

void NeuralNetwork::save(std::string path) {
    std::ofstream checkpoint(path);
    if(checkpoint.is_open()) {
        for(unsigned int l = 0; l < layers.size(); l++) {
            for(unsigned int n = 0; n < layers[l].out_features(); n++) {
                for(unsigned int i = 0; i < layers[l].in_features(); i++)
                    checkpoint << layers[l].node(n)->weight(i) << " ";

                checkpoint << layers[l].node(n)->bias() << "\n";
            }
        }
        checkpoint.close();
    }
}

void NeuralNetwork::load(std::string path) {
    std::ifstream checkpoint(path);
    if(checkpoint.is_open()) {
        std::string line = "";
        for(unsigned int l = 0; l < layers.size(); l++) {
            for(unsigned int n = 0; n < layers[l].out_features(); n++) {
                std::getline(checkpoint, line);

                unsigned int start = 0;
                for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                    double weight = std::stod(line.substr(start, static_cast<unsigned int>(line.find(" ")) - start));
                    layers[l].node(n)->set_weight(i, weight);

                    start = static_cast<unsigned int>(line.find(" ")) + 1;
                }

                double bias = std::stod(line.substr(start, line.length() - start));
                layers[l].node(n)->set_bias(bias);
            }
        }

        checkpoint.close();
    }
}

