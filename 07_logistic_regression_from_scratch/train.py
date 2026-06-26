from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent / "data" / "admissions_generated.csv"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def main():
    data = np.genfromtxt(DATA_PATH, delimiter=",", names=True)

    X = np.column_stack(
        [
            data["study_hours"].astype(float),
            data["practice_tests"].astype(float),
        ]
    )
    y = data["admitted"].astype(float).reshape(-1, 1)

    split_index = int(len(X) * 0.8)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    X_train_scaled = (X_train - feature_mean) / feature_std
    X_test_scaled = (X_test - feature_mean) / feature_std

    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros((n_features, 1))
    bias = 0.0
    learning_rate = 0.1
    epochs = 3_000

    for epoch in range(epochs):
        logits = X_train_scaled @ weights + bias
        probabilities = sigmoid(logits)
        loss = binary_cross_entropy(y_train, probabilities)

        errors = probabilities - y_train
        d_weights = (X_train_scaled.T @ errors) / n_samples
        d_bias = np.sum(errors) / n_samples

        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias

        if epoch % 300 == 0 or epoch == epochs - 1:
            train_predictions = (probabilities >= 0.5).astype(int)
            train_accuracy = np.mean(train_predictions == y_train)
            print(f"epoch {epoch:4d} | loss {loss:.4f} | train accuracy {train_accuracy:.2f}")

    test_probabilities = sigmoid(X_test_scaled @ weights + bias)
    test_predictions = (test_probabilities >= 0.5).astype(int)
    test_accuracy = np.mean(test_predictions == y_test)

    print()
    print("Learned model")
    print(f"scaled weights: {weights.ravel()}")
    print(f"scaled bias:    {bias:.3f}")
    print()
    print("Test predictions")
    for features, probability, predicted, actual in zip(
        X_test,
        test_probabilities,
        test_predictions,
        y_test,
    ):
        hours, tests = features
        print(
            f"{hours:3.1f} study hours, {tests:.0f} practice tests -> "
            f"probability={probability.item():.3f}, "
            f"predicted={predicted.item()}, "
            f"actual={int(actual.item())}"
        )

    print()
    print(f"Test accuracy: {test_accuracy:.2f}")

    examples = np.array(
        [
            [2.0, 1],
            [4.5, 3],
            [6.0, 5],
            [8.0, 6],
        ],
        dtype=float,
    )
    examples_scaled = (examples - feature_mean) / feature_std
    example_probabilities = sigmoid(examples_scaled @ weights + bias)

    print()
    print("New applicant predictions")
    for features, probability in zip(examples, example_probabilities):
        label = "admitted" if probability.item() >= 0.5 else "not admitted"
        print(
            f"{features[0]:3.1f} study hours, {features[1]:.0f} practice tests -> "
            f"{label} ({probability.item():.3f})"
        )


if __name__ == "__main__":
    main()
