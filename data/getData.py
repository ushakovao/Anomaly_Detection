import yfinance as yf
import requests
import pandas as pd
import pandas_datareader.data as web
import datetime


ALPHA_VANTAGE_API_KEY = "ADP45J3Q2C9U3FUE"


# --- 1. Yahoo Finance Data ---
def get_yahoo_finance_data():

    gme_data = yf.download("GME", start="2020-01-01", end="2021-12-31", interval="1d")
    gme_data.columns = [col[0] for col in gme_data.columns]
    gme_data.reset_index(inplace=True)
    gme_data.to_csv("GME_yahoo_data.csv", index=False)

    print("Yahoo Finance Data Collected")
    print("Columns in gme_yahoo:", gme_data.columns)
    return gme_data


# --- 2. Alpha Vantage Data ---
def get_alpha_vantage_data():
    symbol = "GME"
    interval = "5min"
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
    response = requests.get(url)
    data = response.json()

    if "Time Series (5min)" in data:

        df = pd.DataFrame.from_dict(data['Time Series (5min)'], orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        df.to_csv("GME_alpha_vantage_data.csv", index=False)
        print("Columns in alpha:", df.columns)
        print("Alpha Vantage Data Collected")
        return df
    else:
        print("Error: Alpha Vantage API response is not valid")
        return None


# --- 3. FRED Data (Volatility Index) ---
def get_fred_data():
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2021, 12, 31)

    # Volatility Index (VIX) data from FRED
    vix_data = web.DataReader("VIXCLS", "fred", start, end)
    vix_data.to_csv("VIX_fred_data.csv")
    print("FRED Data Collected (VIX)")
    return vix_data



if __name__ == "__main__":
    # 1. Yahoo Finance Data
    get_yahoo_finance_data()

    # 2. Alpha Vantage Data
    get_alpha_vantage_data()

    # 3. FRED Data (VIX)
    get_fred_data()

    print("All Data Collection Completed!")
