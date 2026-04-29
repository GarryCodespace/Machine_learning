import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# -----------------------------
# 1. DATA
# -----------------------------
# Features: [size, rooms]
X = np.array([
    [100, 3],
    [150, 4],
    [200, 5],
    [250, 6]
], dtype=float)

# Prices
y = np.array([300, 500, 700, 900], dtype=float)


# =============================
# PART 1 — NUMPY (FROM SCRATCH)
# =============================

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

n_samples, n_features = X_norm.shape

# Initialize
weights = np.zeros(n_features)
bias = 0

learning_rate = 0.01

# times repeated
epochs = 10000

# Training loop
for epoch in range(epochs):
    y_pred_np = np.dot(X_norm, weights) + bias
    error = y_pred_np - y

    dw = (1 / n_samples) * np.dot(X_norm.T, error)
    db = (1 / n_samples) * np.sum(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db


# Prediction function (NumPy)
def predict_numpy(size, rooms):
    x = np.array([size, rooms])
    x = (x - X_mean) / X_std
    return np.dot(x, weights) + bias


print("NumPy Prediction:", predict_numpy(180, 4))


# =============================
# PART 2 — SCIKIT-LEARN
# =============================

model = LinearRegression()
model.fit(X, y)

pred_sklearn = model.predict([[180, 4]])
print("Sklearn Prediction:", pred_sklearn[0])


# =============================
# PART 3 — ERROR
# =============================

preds = model.predict(X)
mse = mean_squared_error(y, preds)

print("MSE:", mse)


# =============================
# PART 4 — PLOT
# =============================

plt.scatter(y, preds)
plt.xlabel("Real Prices")
plt.ylabel("Predicted Prices")
plt.title("Real vs Predicted")

# Perfect prediction line
plt.plot([min(y), max(y)], [min(y), max(y)])

plt.show()