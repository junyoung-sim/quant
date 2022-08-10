
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>

#include "../lib/quant.hpp"

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
                    double weight = std::stod(line.substr(0, static_cast<unsigned int>(line.find(" "))));
                    agent.layer(l)->node(n)->set_weight(i, weight);

                    line = line.substr(static_cast<unsigned int>(line.find(" ")) + 1);
                }

                double bias = std::stod(line);
                agent.layer(l)->node(n)->set_bias(bias);
            }
        }

        sync();
        out.close();
    }
 }
