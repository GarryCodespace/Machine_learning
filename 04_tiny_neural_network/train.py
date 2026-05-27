import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# binary cross entropy loss (log loss)
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
# clip value between [0.000000001,0.999999999]
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def main():
    # XOR data. The model sees two inputs and learns to predict 0 or 1.
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    y = np.array([[0], [1], [1], [0]], dtype=float)

#generaet random weights
    rng = np.random.default_rng(seed=42)

# XOR has 2 inputs
    input_size = 2

# 4 neurons inbetween inputs and output
    hidden_size = 4
    output_size = 1

#learning rate adjusting the gradient weights
    learning_rate = 0.3

# number repeated
    epochs = 10_000

    # Layer 1: input -> hidden 2*4
    W1 = rng.normal(0, 1, size=(input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))

    # Layer 2: hidden -> output, 4 * 1
    W2 = rng.normal(0, 1, size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))

    for epoch in range(epochs):
        # Forward pass
    # calcuating the output 
        z1 = X @ W1 + b1
    # make the fucntion nonlinear
        hidden = np.tanh(z1)
    # hidden to output
        z2 = hidden @ W2 + b2
        predictions = sigmoid(z2)

        loss = binary_cross_entropy(y, predictions)

        # Backward pass
        n_samples = len(X)
    
        # average loss over 4 samples
        dz2 = (predictions - y ) / n_samples
        dW2 = hidden.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)


        dhidden = dz2 @ W2.T
# differentiate tan
        dz1 = dhidden * (1 - hidden**2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)    

        # Gradient descent update
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:5d} | loss {loss:.4f}")

    final_probabilities = predictions
    final_classes = (final_probabilities >= 0.5).astype(int)
    accuracy = np.mean(final_classes == y)

    print()
    print("Final predictions")
    for inputs, probability, predicted, target in zip(X, final_probabilities, final_classes, y):
        print(
            f"{inputs.astype(int).tolist()} -> "
            f"probability={probability[0]:.3f}, "
            f"predicted={predicted[0]}, "
            f"target={int(target[0])}"
        )

    print()
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
