# Logistic Regression From Scratch: Chat Notes

Source: https://chatgpt.com/share/6a1e9eed-6a50-83ec-8393-3146719157bd

Related local notebook: `/Users/garryyuan/machine_learning/logistic_regression_chat_notes.ipynb`

## Summary

These notes capture the main learning points from the chat about a NumPy logistic regression script for an admissions dataset. The discussion covered NumPy vs pandas, loading CSV data, `Path`, sigmoid, binary cross-entropy, averaging loss with `1/N`, feature and label arrays, shape unpacking, logits, gradient descent, model evaluation, and the next step of learning CIFAR-10 CNNs.

The core example predicts whether a student is admitted using:

- `study_hours`
- `practice_tests`

Target:

- `admitted`

## NumPy vs Pandas

The original code used NumPy:

```python
data = np.genfromtxt(DATA_PATH, delimiter=",", names=True)
```

This is fine because the dataset is small, numeric, and simple. NumPy can load the CSV, select columns, and perform all the matrix math needed for logistic regression.

Pandas could also be used:

```python
import pandas as pd

data = pd.read_csv(DATA_PATH)

X = data[["study_hours", "practice_tests"]].to_numpy()
y = data["admitted"].to_numpy().reshape(-1, 1)
```

Practical rule:

- Use pandas for loading, cleaning, filtering, grouping, joining, and inspecting data.
- Use NumPy arrays for the model math once the data is ready.

## Path and DATA_PATH

The original script used:

```python
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "admissions_generated.csv"
```

Meaning:

- `Path` helps build file paths cleanly.
- `__file__` is the current Python script.
- `.parent` gets the folder containing that script.
- `/` joins path parts.

If the project is:

```text
project/
├── train.py
└── data/
    └── admissions_generated.csv
```

then `DATA_PATH` points to:

```text
project/data/admissions_generated.csv
```

In notebooks, `__file__` is usually not available, so `Path.cwd()` is often used instead.

## Sigmoid

The sigmoid function is:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Mathematically:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

It converts any real number into a probability between `0` and `1`.

- Large positive `z` gives a probability near `1`.
- Large negative `z` gives a probability near `0`.
- `z = 0` gives `0.5`.

## What `admitted` Means

`admitted` is the target variable, also called the label or output.

Example:

| study_hours | practice_tests | admitted |
|---:|---:|---:|
| 2 | 1 | 0 |
| 5 | 3 | 1 |
| 8 | 4 | 1 |

Usually:

```text
admitted = 1 means yes, admitted
admitted = 0 means no, not admitted
```

The model learns:

```text
study_hours + practice_tests -> admitted
```

## Building X and y

The code:

```python
data = np.genfromtxt(DATA_PATH, delimiter=",", names=True)

X = np.column_stack(
    [
        data["study_hours"].astype(float),
        data["practice_tests"].astype(float),
    ]
)
y = data["admitted"].astype(float).reshape(-1, 1)
```

`X` is the feature matrix:

```text
rows = students
columns = study_hours and practice_tests
```

`y` is the answer column:

```text
admitted
```

`.reshape(-1, 1)` turns a flat array into a column vector.

## Shape and Tuple Unpacking

This line:

```python
n_samples, n_features = X_train_scaled.shape
```

defines two variables.

If:

```python
X_train_scaled.shape == (800, 2)
```

then:

```python
n_samples = 800
n_features = 2
```

This is called tuple unpacking.

Why use it?

The model needs one weight per feature:

```python
weights = np.zeros((n_features, 1))
```

If there are two features, the weights start as:

```python
[[0.],
 [0.]]
```

## Weights, Bias, Learning Rate, Epochs

Setup:

```python
weights = np.zeros((n_features, 1))
bias = 0.0
learning_rate = 0.1
epochs = 3_000
```

Meaning:

- `weights`: learned importance of each feature.
- `bias`: intercept added to the model.
- `learning_rate`: how big each gradient descent update is.
- `epochs`: how many training iterations to run.

## Logit

The logit is the raw linear score before sigmoid.

With two features:

```text
z = w1*x1 + w2*x2 + b
```

This is the same idea as:

```text
ax1 + bx2 + c
```

In code:

```python
logits = X_train_scaled @ weights + bias
```

The `@` operator means matrix multiplication, so NumPy computes the linear expression for every row at once.

Pipeline:

```text
features -> logit -> sigmoid -> probability -> prediction
```

## Binary Cross-Entropy

Function:

```python
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )
```

Binary cross-entropy measures how bad the model's predicted probabilities are.

Formula:

```text
L = -(1/N) * sum(y_i * log(yhat_i) + (1 - y_i) * log(1 - yhat_i))
```

Where:

- `y_i` is the real label, `0` or `1`.
- `yhat_i` is the predicted probability.
- `N` is the number of examples.

The `epsilon` prevents `log(0)`, which is undefined.

## Why Use 1/N?

The `1/N` averages the loss across all examples.

Without averaging, a dataset with 1000 rows would naturally have about 10 times more total loss than a dataset with 100 rows, even if the model quality were the same.

Averaging makes the loss comparable across dataset sizes.

In code:

```python
np.mean(...)
```

does the averaging.

## Training Loop

Core loop:

```python
for epoch in range(epochs):
    logits = X_train_scaled @ weights + bias
    probabilities = sigmoid(logits)
    loss = binary_cross_entropy(y_train, probabilities)

    errors = probabilities - y_train
    d_weights = (X_train_scaled.T @ errors) / n_samples
    d_bias = np.sum(errors) / n_samples

    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias
```

Conceptually:

1. Make predictions.
2. Measure loss.
3. Compute errors.
4. Compute gradients.
5. Update weights and bias.
6. Repeat.

The gradient tells the model how to adjust parameters to reduce loss.

## Evaluation

After training:

```python
test_probabilities = sigmoid(X_test_scaled @ weights + bias)
test_predictions = (test_probabilities >= 0.5).astype(int)
test_accuracy = np.mean(test_predictions == y_test)
```

Meaning:

- Compute probabilities on unseen test data.
- Convert probabilities to `0` or `1` using a threshold of `0.5`.
- Compare predictions to the true labels.

Rule:

```text
probability >= 0.5 -> predict 1
probability < 0.5  -> predict 0
```

## CIFAR-10 CNN Next Step

The chat also discussed this recommendation:

> Start with CIFAR-10 CNN after basic MNIST.

CIFAR-10 is a classic image classification dataset:

- 60,000 color images
- 10 classes
- 32 by 32 pixels

Classes:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

It is harder than MNIST because CIFAR-10 has:

- color images
- real-world objects
- varied backgrounds
- different object positions

It teaches:

- `torchvision.datasets`
- data augmentation
- `Conv2d`
- `MaxPool2d`
- `BatchNorm2d`
- better training loops
- architecture experiments

Suggested learning path:

1. Logistic regression from scratch with NumPy.
2. Basic neural network on MNIST.
3. CNN on CIFAR-10.
4. Deeper CNNs, augmentation, schedulers, and regularization.

## Glossary

| Term | Meaning |
|---|---|
| Feature | Input column used for prediction, such as `study_hours`. |
| Target / label | Correct answer the model learns to predict, such as `admitted`. |
| `X` | Feature matrix. Rows are examples, columns are features. |
| `y` | Target values. Usually shaped as a column vector. |
| Weight | Learned coefficient for a feature. |
| Bias | Learned intercept added to the linear score. |
| Logit | Raw linear score: `X @ weights + bias`. |
| Sigmoid | Function that converts logits into probabilities. |
| Probability | Model confidence between `0` and `1`. |
| Prediction | Final class after thresholding probability. |
| BCE loss | Binary cross-entropy, used for binary classification. |
| Gradient | Direction showing how to change parameters to reduce loss. |
| Learning rate | Step size for parameter updates. |
| Epoch | One training iteration cycle in this simple loop. |
| Test set | Data held out to evaluate generalization. |

## Key Takeaways

- The original code did not need pandas because NumPy can load simple numeric CSVs and do the matrix math.
- Pandas is still a good choice for real-world data loading and inspection.
- `n_samples, n_features = X.shape` defines two variables through tuple unpacking.
- Logistic regression computes a linear score first.
- The linear score is called the logit.
- Sigmoid converts the logit into a probability.
- Binary cross-entropy rewards confident correct probabilities and punishes confident wrong probabilities.
- The `1/N` average keeps loss comparable across dataset sizes.
- Gradient descent updates weights and bias a little each epoch.
- Test accuracy checks whether the model works on unseen data.
- CIFAR-10 CNNs are a strong next project after MNIST.
