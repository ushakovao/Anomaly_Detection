import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

gme_data = yf.download("GME", start="2020-01-01", end="2021-12-01")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = np.array(range(len(gme_data))).reshape(-1, 1)
y = gme_data['Close'].values.reshape(-1, 1)
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

gp.fit(X_scaled, y_scaled)

y_mean, y_std = gp.predict(X_scaled, return_std=True)

residuals = y_scaled.flatten() - y_mean.flatten()

threshold_factor = 3
threshold = threshold_factor * y_std

percentage_deviation = np.abs(residuals) / np.abs(y_scaled.flatten())

percentage_threshold = 0.1
anomalies = (np.abs(residuals) > threshold) & (percentage_deviation > percentage_threshold)


anomaly_indices = np.where(anomalies)[0]

print("Predictions vs Actual:")
for idx in anomaly_indices:
    predicted_price = scaler_y.inverse_transform(y_mean[idx].reshape(1, -1))[0][0]
    actual_price = gme_data['Close'].iloc[idx]
    price_diff = predicted_price - actual_price.values[0]
    print(f"Date: {gme_data.index[idx].strftime('%Y-%m-%d')}, Predicted: {predicted_price:.2f}, Actual: {actual_price.values[0]:.2f}, Difference: {price_diff:.2f}")

print(f"Detected {np.sum(anomalies)} anomalies")

plot_data = gme_data[(gme_data.index >= '2021-01-01') & (gme_data.index <= '2021-07-01')]

plot_indices = (gme_data.index >= '2021-01-01') & (gme_data.index <= '2021-07-01')

plot_X_scaled = scaler_X.transform(np.array(range(len(gme_data)))[plot_indices].reshape(-1, 1))
plot_y_mean, _ = gp.predict(plot_X_scaled, return_std=True)
plot_y_mean = scaler_y.inverse_transform(plot_y_mean.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(plot_data.index, plot_data['Close'], label="Original Data", color="lightblue", zorder=2)
plt.scatter(gme_data.index[anomalies & plot_indices], gme_data['Close'][anomalies & plot_indices], color="yellow", label="Anomalies", zorder=3)
plt.plot(plot_data.index, plot_y_mean, label="GP Prediction", color="red", linestyle="--", zorder=4)
plt.title("Stock Price with Anomalies and GP Predictions (2021-01-01 to 2021-07-01)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()

plt.tight_layout()
plt.show()
