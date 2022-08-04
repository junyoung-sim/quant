
#include <cstdlib>
#include <vector>
#include <string>

#include <iostream>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::vector<std::string> tickers = {"SPY", "TLT", "GSG", "^VIX"};
    Market market(tickers);

    return 0;
}