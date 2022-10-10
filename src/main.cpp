#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "../lib/quant.hpp"

std::string cmd = "";
std::string mode = "";
std::vector<std::string> tickers;
std::string checkpoint = "";
std::vector<Market> dataset;
Quant *quant;

#define VALID_MODE (mode == "build" || mode == "run")
#define VALID_CHECKPOINT (checkpoint.find("./models/") == 0)

void boot(int argc, char *argv[]) {
    unsigned int arg = 1;
    mode = argv[arg++];
    while(arg <= argc - 2)
        tickers.push_back(argv[arg++]);
    if(arg == argc - 1)
        checkpoint = argv[arg];
}

int terminate() {
    std::vector<std::string>().swap(tickers);
    std::vector<Market>().swap(dataset);
    delete quant;

    return 0;
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    if(VALID_MODE && VALID_CHECKPOINT) {
        std::cout << "Downloading SPY ^TNX IEF XLE\n";
        cmd = "./python/download.py SPY ^TNX IEF XLE";
        std::system(cmd.c_str());

        for(std::string &ticker: tickers) {
            std::cout << "Downloading " << ticker << "\n";
            cmd = "./python/download.py " + ticker + " && ./python/clean.py " + ticker + " SPY ^TNX IEF XLE";
            std::system(cmd.c_str());

            dataset.push_back(Market({ticker, "SPY", "^TNX", "IEF", "XLE"}));
        }

        quant = new Quant(dataset, checkpoint);

        if(mode == "build") quant->build();
        else quant->run();
    }
    else
        std::cout << "Boot-time failure!\n";

    terminate();
}