#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

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
    // clean historical data using pandas: columns={ticker, indicators}
    std::string clean = "./python/clean.py " + ticker + " ";
    for(unsigned int i = 0; i < indicators.size(); i++) {
        clean += indicators[i];
        if(i < indicators.size() - 1)
            clean += " ";
    }
    std::system(clean.c_str());

    // each row of 2d vec has the historical data of the ticker and the indicators
    return read_csv("./data/cleaned.csv");
}

double mean(std::vector<double> &dat) {
    double sum = 0.00;
    for(double &x: dat)
        sum += x;
    return sum / dat.size();
}

double stdev(std::vector<double> &dat) {
    double s = 0.00;
    double m = mean(dat);
    for(double &x: dat)
        s += pow(x - m, 2);
    s /= dat.size();
    s = sqrt(s);
    return s;
}

void piecewise_aggregate_approximation(std::vector<double> &dat, unsigned int window) {
    // discretizing time series by averaging equally split windows
    for(unsigned int t = 0; t <= dat.size() - window; t += window) {
        std::vector<double> piece = {dat.begin() + t, dat.begin() + t + window};
        double approx = mean(piece);
        for(unsigned int i = t; i < t + window; i++)
            dat[i] = approx;
    }
}

void standardize(std::vector<double> &dat) {
    double m = mean(dat);
    double s = stdev(dat);
    for(double &x: dat)
        x = (x - m) / s;
}