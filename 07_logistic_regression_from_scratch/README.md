# NumPy Logistic Regression From Scratch

Goal: learn binary classification without `sklearn`, PyTorch, or `.fit()`.

This project predicts whether a student is admitted from two features:

```text
study_hours, practice_tests -> admitted
```

The model learns:

```text
probability = sigmoid(features @ weights + bias)
```

## Run

```bash
python 07_logistic_regression_from_scratch/train.py
```

## What To Interrogate

- Why does logistic regression use `sigmoid`?
- What is a probability threshold?
- Why does this use binary cross-entropy?
- What are `weights` and `bias`?
- What does `probabilities - y_train` mean?
- How is this similar to the neural network output layer?

## The Core Training Loop

```python
logits = X_train_scaled @ weights + bias
probabilities = sigmoid(logits)
errors = probabilities - y_train

d_weights = (X_train_scaled.T @ errors) / n_samples
d_bias = np.sum(errors) / n_samples

weights -= learning_rate * d_weights
bias -= learning_rate * d_bias
```
