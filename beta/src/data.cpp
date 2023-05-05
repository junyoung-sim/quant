#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>

void download(std::vector<std::string> &tickers) {
    for(std::string &x: tickers)
        std::system(("./python/download.py " + x).c_str());
}

std::vector<std::vector<double>> read_csv(std::string path) {
    std::vector<std::vector<double>> dat;
    std::ifstream file(path);

    if(file.is_open()) {
        std::string line, val;
        std::getline(file, line);

        unsigned int columns = 1;
        for(char &ch: line)
            columns += (ch == ',');

        dat.resize(columns, std::vector<double>());
        while(std::getline(file, line)) {
            for(unsigned int col = 0; col < columns; col++) {
                double val = std::stod(line.substr(0, line.find(",")));
                dat[col].push_back(val);

                line = line.substr(line.find(",") + 1);
            }
        }

        file.close();
    }

    return dat;
}

std::vector<std::vector<double>> historical_data(std::string ticker, std::vector<std::string> &indicators) {
    std::string clean = "./python/clean.py " + ticker + " ";
    for(unsigned int i = 0; i < indicators.size(); i++) {
        clean += indicators[i];
        if(i < indicators.size() - 1)
            clean += " ";
    }
    std::system(clean.c_str());

    return read_csv("./data/cleaned.csv");
}