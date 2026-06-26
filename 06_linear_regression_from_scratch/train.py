from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).parent / "data" / "houses_generated.csv"


def mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def main():
    # load the data from the CSV file
    data = np.genfromtxt(DATA_PATH, delimiter=",", names=True)

    x = data["size_sqft"].astype(float)
    y = data["price_k"].astype(float)

    split_index = int(len(x) * 0.8)
    # takes first 80% of the data for training and the rest for testing
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]

    x_mean = x_train.mean()
    x_std = x_train.std()
    x_train_scaled = (x_train - x_mean) / x_std
    x_test_scaled = (x_test - x_mean) / x_std

    weight = 0.0
    bias = 0.0
    learning_rate = 0.05
    epochs = 2_000
    n_samples = len(x_train_scaled)

    for epoch in range(epochs):
        predictions = weight * x_train_scaled + bias
        errors = predictions - y_train
        loss = mean_squared_error(y_train, predictions)

        d_weight = (2 / n_samples) * np.sum(errors * x_train_scaled)
        d_bias = (2 / n_samples) * np.sum(errors)

        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias
 
        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d} | train MSE {loss:.2f}")

    # test the model on the test set
    test_predictions = weight * x_test_scaled + bias
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = np.mean(np.abs(test_predictions - y_test))
    # convert the scaled model back to the original units
    raw_slope = weight / x_std
    raw_intercept = bias - (weight * x_mean / x_std)

    print()
    print("Learned model")
    print(f"scaled equation: price_k = {weight:.2f} * scaled_size + {bias:.2f}")
    print(f"raw equation:    price_k = {raw_slope:.3f} * size_sqft + {raw_intercept:.2f}")
    print()
    print("Test metrics")
    print(f"MSE: {test_mse:.2f}")
    print(f"MAE: ${test_mae:.2f}k")
    print()
    print("Example test predictions")
    for size, actual, predicted in zip(x_test, y_test, test_predictions):
        print(f"{size:4.0f} sqft -> predicted ${predicted:7.1f}k | actual ${actual:7.1f}k")

    example_sizes = np.array([900, 1600, 2400, 3200], dtype=float)
    example_sizes_scaled = (example_sizes - x_mean) / x_std
    example_prices = weight * example_sizes_scaled + bias

    print()
    print("New house predictions")
    for size, price in zip(example_sizes, example_prices):
        print(f"{size:4.0f} sqft -> predicted ${price:.1f}k")


if __name__ == "__main__":
    main()
