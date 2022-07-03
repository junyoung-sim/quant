
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "../lib/data.hpp"
#include "../lib/dqn.hpp"

int main(int argc, char *argv[])
{
    std::vector<double> series = read_csv("./data/SPXL_1min.csv", "Close");

    

    return 0;
}
