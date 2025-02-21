Anomaly Detection in Multivariate Financial Time Series


Generated Files:

GME_yahoo_data.csv:
    Historical daily stock data for GameStop (GME), including Open, High, Low, Close prices and Volume for each trading day from 2020 to 2021.
GME_alpha_vantage_data.csv:
    Intraday stock data for GameStop (GME) at 5-minute intervals, including Open, High, Low, Close prices and Volume for 2021.
VIX_fred_data.csv:
    Volatility Index (VIX) data from FRED, representing overall market volatility, from 2020 to 2021.



IA models to  implement:

Deep Autoencoders:
Use Deep Autoencoders to model the normal behavior of GameStop’s stock data (price and volume) over time. 
Once trained, any significant deviation from the normal data will be flagged as an anomaly.

Input: Normalized stock prices, trading volumes and possibly the VIX data.
Output: Anomalies (e.g., unusual price movements, abnormal volume spikes).

Gaussian Process Models (GPM):
Gaussian Process Models to detect non-linear trends and uncertainties in time-series data. 
GPM is used to model the distribution of stock prices.
Anomalies can be detected when the observed data points fall far from the predicted distribution.

Input: Historical stock prices and volumes, potentially augmented with VIX data as a feature.
Output: Predicted values vs. observed values. Anomalies are detected when observed data deviates significantly from predictions.