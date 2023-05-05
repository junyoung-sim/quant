#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "../lib/data.hpp"

std::string mode; // build, test, run
std::string checkpoint; // model path (./models/*)

std::map<std::string, std::vector<std::vector<double>>> environment; // (ticker, historical data)

/*
    ./exec mode <tickers> checkpoint
*/

void boot(int argc, char *argv[]) {
    mode = argv[1];
    checkpoint = argv[argc-1];

    // read ticker list from command-line
    std::vector<std::string> tickers;
    std::vector<std::string> indicators = {"SPY", "IEF", "EUR=X", "GSG"};
    for(unsigned int i = 2; i < argc-1; i++)
        tickers.push_back(argv[i]);
    
    // download historical data
    std::cout << "\nDownloading... (this may take a while)\n";
    download(tickers);
    download(indicators);

    // create environment for each ticker
    for(std::string &x: tickers)
        environment[x] = historical_data(x, indicators);
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    

    return 0;
}