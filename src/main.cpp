
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "../lib/dqn.hpp"

std::vector<std::vector<double>> read(std::string path) {
    std::ifstream file(path);
    std::vector<std::vector<double>> dat;

    if(file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            if(line != "\n") {
                std::vector<double> row;
                unsigned int start = 0;
                for(unsigned int i = 0; i < line.length(); i++) {
                    if(line[i] == ' ' || i == line.length() - 1) {
                        row.push_back(std::stod(line.substr(start, i - start)));
                        start = i + 1;
                    }
                }
                dat.push_back(row);
                std::vector<double>().swap(row);
            }
        }

        file.close();
    }

    return dat;
}

int main(int argc, char *argv[])
{
    std::vector<std::vector<double>> state = read("./data/residual");
    std::vector<std::vector<double>> reward = read("./data/reward");

    DQN dqn({{30,20},{20,10},{10,5},{5,2}});
    dqn.optimize(state, reward);

    std::vector<std::vector<double>>().swap(state);
    std::vector<std::vector<double>>().swap(reward);

    return 0;
}
