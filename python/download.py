#!/usr/bin/env python3

import sys
import certifi
import warnings
import json, csv
from urllib.request import urlopen

warnings.filterwarnings("ignore")

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

if __name__ == "__main__":
    ticker = sys.argv[1]
    apikey = open("./apikey", "r").readline()

    url = "https://financialmodelingprep.com/api/v3/historical-price-full"
    if ticker.endswith("=X"):
        url += "/{}?apikey={}&from=2000-01-01" .format("USD" + ticker[:-2], apikey)
    else:
        url += "/{}?apikey={}&from=2000-01-01" .format(ticker, apikey)

    json_data = get_jsonparsed_data(url)
    out = open("./data/{}.csv" .format(ticker), "w")
    csv_writer = csv.writer(out)

    header = True
    for data in json_data["historical"]:
        if header:
            csv_writer.writerow(data.keys())
            header = False
        csv_writer.writerow(data.values())
    out.close()