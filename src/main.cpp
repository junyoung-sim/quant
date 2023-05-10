#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>

#include "../lib/quant.hpp"

std::string mode; // build, test, run
std::string checkpoint; // model path (./models/*)

std::vector<std::string> tickers; // tickers of interest
std::vector<std::string> indicators = {"SPY", "IEF", "EUR=X", "GSG"}; // stock bond currency commodities
Environment env; // (ticker, historical data)

/*
    ./exec mode <tickers> checkpoint
*/

void boot(int argc, char *argv[]) {
    mode = argv[1];
    checkpoint = argv[argc-1];

    // read ticker list from command-line
    for(unsigned int i = 2; i < argc-1; i++)
        tickers.push_back(argv[i]);
    
    // download historical data
    std::cout << "\nDownloading... (this may take a while)\n";
    download(tickers);
    download(indicators);

    // create environment for each ticker
    for(std::string &ticker: tickers)
        env[ticker] = historical_data(ticker, indicators);

    std::cout << "\n";
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    Quant quant(checkpoint);

    if(mode == "build")
        quant.build(tickers, env);
    else if(mode == "test")
        quant.test(tickers, env);
    else if(mode == "run")
        quant.run(tickers, env);
    else
        std::cout << "Invalid mode given.\n";

    return 0;
}