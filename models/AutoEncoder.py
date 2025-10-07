import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

gme_data = yf.download("GME", start="2020-01-01", end="2021-12-01")
dates = gme_data.index
prices = gme_data['Close'].values.reshape(-1, 1)

split_date = '2021-01-01'
train_mask = dates < split_date
test_mask = dates >= split_date
y_train = prices[train_mask]
y_test = prices[test_mask]

scaler = StandardScaler()
y_train_s = scaler.fit_transform(y_train)
y_test_s = scaler.transform(y_test)

window_size = 10


def make_windows(data):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)


X_train = make_windows(y_train_s)
X_test = make_windows(y_test_s)

y_train_w = X_train[:, -1, 0].reshape(-1, 1)
y_test_w = X_test[:, -1, 0].reshape(-1, 1)

model = Sequential([
    Flatten(input_shape=(window_size, 1)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(window_size, activation='linear'),
    Reshape((window_size, 1)),
])
model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, X_train,
    epochs=50, batch_size=16, verbose=0
)
print(f"Final training loss: {history.history['loss'][-1]:.6f}")

X_test_recon = model.predict(X_test)
pred_scaled = X_test_recon[:, -1, 0].reshape(-1, 1)
pred = scaler.inverse_transform(pred_scaled)

actual = y_test[window_size:]

residuals = (actual.flatten() - pred.flatten())
pct_dev = np.abs(residuals) / actual.flatten()

thr_abs = 3 * np.std(residuals)
thr_pct = 0.10

anomalies = (np.abs(residuals) > thr_abs) & (pct_dev > thr_pct)
print(f"Anomalies in 2021: {anomalies.sum()} out of {len(residuals)} points")

test_indices = np.where(test_mask)[0]
plot_indices = test_indices[window_size:]

start, end = '2021-01-01', '2021-07-01'
mask_plot = (dates >= start) & (dates <= end)

plt.figure(figsize=(10, 6))
plt.plot(dates[mask_plot], prices[mask_plot], color='lightblue', label='Original Data', zorder=2)
pred_dates = dates[plot_indices]
pred_mask = (pred_dates >= start) & (pred_dates <= end)
plt.plot(pred_dates[pred_mask], pred[pred_mask], '--', color='red', label='AutoEncoder Prediction', zorder=4)
anom_idx = plot_indices[anomalies]
anom_mask = (dates[anom_idx] >= start) & (dates[anom_idx] <= end)
plt.scatter(dates[anom_idx][anom_mask], prices[anom_idx][anom_mask], color='yellow', label='Anomalies', zorder=3)

plt.title(f"GME Price with Windowed AutoEncoder Anomalies ({start} to {end})")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.show()
