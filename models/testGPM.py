import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import poisson

np.random.seed(42)

time_points = np.arange(500)

initial_price = 100
lognormal_returns = np.random.normal(loc=0, scale=0.01, size=len(time_points))

price_changes = np.cumsum(lognormal_returns)
data = initial_price * np.exp(price_changes)

poisson_jumps = poisson.rvs(mu=0.1, size=len(time_points))
jump_sizes = np.random.normal(loc=5, scale=2, size=len(time_points)) * poisson_jumps
data += jump_sizes

anomaly_indices = np.random.choice(len(time_points), size=5, replace=False)
data[anomaly_indices] += np.random.normal(loc=50, scale=10, size=5)

kernel = C(1.0, (1e-4, 1e2)) * RBF(30, (1e-4, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                              alpha=1e-1)

gp.fit(time_points.reshape(-1, 1), data)

predictions, sigma = gp.predict(time_points.reshape(-1, 1), return_std=True)

z_scores = np.abs((data - predictions) / sigma)
threshold_z = 2

anomalies_z = z_scores > threshold_z
num_anomalies_z = np.sum(anomalies_z)

print(f"Number of detected anomalies: {num_anomalies_z}")

plt.figure(figsize=(10, 6))

plt.plot(time_points, data, label='Generated Data (Stock Prices)', color='blue', alpha=0.7)

plt.plot(time_points, predictions, label='GPM Predictions', color='red', linestyle='--')

plt.scatter(time_points[anomalies_z], data[anomalies_z], color='green', label='Anomalies', zorder=5)

plt.title("Anomaly Detection using Gaussian Process Model (Z-score method)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
