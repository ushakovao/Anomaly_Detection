# =============================================================
# === ANOMALY DETECTION USING AUTOENCODER & GAUSSIAN PROCESS ===
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# =====================
# ==== PARAMETERS =====
# =====================

# --- Data settings ---
symbol = "GME"              # Stock ticker (GameStop)
start_date = "2020-01-01"   # Start date for data
end_date = "2021-12-01"     # End date for data
split_date = '2021-01-01'   # Date to split training and test data

# --- AutoEncoder parameters ---
ae_window_size = 10          # Number of time steps (rolling window)
ae_epochs = 50               # Number of training epochs
ae_batch_size = 16           # Mini-batch size
thr_abs_ae = 2               # Absolute anomaly threshold (× residual std)
thr_pct_ae = 0.10            # Relative anomaly threshold (10% deviation)

# --- Gaussian Process parameters ---
gp_alpha = 1e-2              # Regularization (noise level)
gp_n_restarts = 10           # Optimization restarts for kernel fitting
thr_pct_gp = 0.10            # Relative anomaly threshold (10% deviation)

# --- Plot settings ---
plot_start = '2021-01-01'
plot_end = '2021-07-01'

# =====================
# ==== DOWNLOAD DATA ===
# =====================
# Download historical stock data from Yahoo Finance
gme_data = yf.download(symbol, start=start_date, end=end_date)

# Extract closing prices and corresponding dates
dates = gme_data.index
prices = gme_data['Close'].values.reshape(-1, 1)

# =====================
# ==== AUTOENCODER ====
# =====================
# Split data into training (before split_date) and testing (after split_date)
train_mask = dates < split_date
test_mask  = dates >= split_date
y_train = prices[train_mask]
y_test  = prices[test_mask]

# Normalize data for stable neural network training
scaler_ae = StandardScaler()
y_train_s = scaler_ae.fit_transform(y_train)
y_test_s  = scaler_ae.transform(y_test)

# Helper function: create overlapping rolling windows
def make_windows(data, window_size):
    return np.array([data[i:i+window_size] for i in range(len(data) - window_size)])

# Create training and test sequences
X_train = make_windows(y_train_s, ae_window_size)
X_test  = make_windows(y_test_s, ae_window_size)

# --- Define AutoEncoder architecture ---
model = Sequential([
    Flatten(input_shape=(ae_window_size,1)),   # Flatten time window
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),              # Bottleneck (compressed representation)
    Dense(64, activation='relu'),
    Dense(ae_window_size, activation='linear'),# Reconstruct the input
    Reshape((ae_window_size,1)),               # Reshape to original shape
])

# Compile and train AutoEncoder
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=ae_epochs, batch_size=ae_batch_size, verbose=0)

# --- Reconstruction on test data ---
X_test_recon = model.predict(X_test)

# Get the last predicted value of each window (forecast-style output)
pred_scaled = X_test_recon[:, -1, 0].reshape(-1,1)
pred = scaler_ae.inverse_transform(pred_scaled)
actual = y_test[ae_window_size:]  # Align actual values

# --- Compute residuals and detect anomalies ---
residuals_ae = (actual.flatten() - pred.flatten())
pct_dev_ae   = np.abs(residuals_ae) / actual.flatten()

# Define thresholds for anomaly detection
thr_abs_val = thr_abs_ae * np.std(residuals_ae)
anomalies_ae = (np.abs(residuals_ae) > thr_abs_val) & (pct_dev_ae > thr_pct_ae)
plot_indices_ae = np.where(test_mask)[0][ae_window_size:]

# =============================
# ==== GAUSSIAN PROCESS =======
# =============================
# Prepare inputs for Gaussian Process regression
X_gp = np.arange(len(gme_data)).reshape(-1, 1)

# Standardize features and targets
scaler_gp_X = StandardScaler()
scaler_gp_y = StandardScaler()
X_scaled = scaler_gp_X.fit_transform(X_gp)
y_scaled = scaler_gp_y.fit_transform(prices)

# Define GP kernel (Constant × RBF)
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

# Train Gaussian Process
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gp_n_restarts, alpha=gp_alpha)
gp.fit(X_scaled, y_scaled)

# Predict mean and standard deviation
y_mean, y_std = gp.predict(X_scaled, return_std=True)

# --- Detect anomalies using deviation from GP mean ---
residuals_gp = y_scaled.flatten() - y_mean.flatten()
threshold_gp = 2 * y_std  # 2-sigma rule
pct_dev_gp_val = np.abs(residuals_gp) / np.abs(y_scaled.flatten())
anomalies_gp = (np.abs(residuals_gp) > threshold_gp) & (pct_dev_gp_val > thr_pct_gp)

# =============================
# ==== COMPARE RESULTS ========
# =============================
# Convert anomaly indices to dates
anom_dates_ae = dates[plot_indices_ae][anomalies_ae]
anom_dates_gp = dates[anomalies_gp]

# Identify overlap and differences
common_anomalies = np.intersect1d(anom_dates_ae, anom_dates_gp)
only_ae = np.setdiff1d(anom_dates_ae, anom_dates_gp)
only_gp = np.setdiff1d(anom_dates_gp, anom_dates_ae)

# Print summary
print("==== Anomaly Detection Comparison ====")
print(f"AutoEncoder anomalies: {len(anom_dates_ae)}")
print(f"Gaussian Process anomalies: {len(anom_dates_gp)}")
print(f"Common anomalies: {len(common_anomalies)}")
print(f"Unique to AE: {len(only_ae)}")
print(f"Unique to GP: {len(only_gp)}")

# =============================
# ==== PRINT PARAMETERS =======
# =============================
print("==== PARAMETERS ====")
print(f"AutoEncoder: window_size={ae_window_size}, epochs={ae_epochs}, batch_size={ae_batch_size}, "
      f"thr_abs_multiplier={thr_abs_ae}, thr_pct={thr_pct_ae}")
print(f"GaussianProcess: alpha={gp_alpha}, n_restarts_optimizer={gp_n_restarts}, thr_pct={thr_pct_gp}")
print("====================\n")

# =============================
# ==== VISUAL COMPARISON ======
# =============================
# Restrict plots to a chosen time window
mask_plot = (dates >= plot_start) & (dates <= plot_end)

plt.figure(figsize=(12, 6))
plt.plot(dates[mask_plot], prices[mask_plot], label='GME Close Price', color='lightblue', zorder=1)

# --- GP prediction ---
plot_idx = np.where(mask_plot)[0]
plot_y_mean = scaler_gp_y.inverse_transform(y_mean.reshape(-1,1))[plot_idx]
plt.plot(dates[mask_plot], plot_y_mean, '--', color='red', label='GP Prediction', zorder=2)

# --- AutoEncoder prediction ---
pred_plot_mask = (dates[plot_indices_ae] >= plot_start) & (dates[plot_indices_ae] <= plot_end)
plt.plot(dates[plot_indices_ae][pred_plot_mask], pred[pred_plot_mask], '--', color='green', label='AE Prediction', zorder=3)

# --- Anomaly markers ---
plt.scatter(anom_dates_gp[(anom_dates_gp >= plot_start) & (anom_dates_gp <= plot_end)],
            gme_data.loc[anom_dates_gp[(anom_dates_gp >= plot_start) & (anom_dates_gp <= plot_end)], 'Close'],
            color='red', label='GP Anomaly', marker='x', zorder=5)

plt.scatter(anom_dates_ae[(anom_dates_ae >= plot_start) & (anom_dates_ae <= plot_end)],
            gme_data.loc[anom_dates_ae[(anom_dates_ae >= plot_start) & (anom_dates_ae <= plot_end)], 'Close'],
            color='green', label='AE Anomaly', zorder=4)

plt.title("AutoEncoder vs GP Anomaly Detection (Jan–Jul 2021)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.show()
