# NumPy Linear Regression From Scratch

Goal: learn what happens under `.fit()` by training linear regression manually.

This project predicts house price from house size:

```text
size_sqft -> price_k
```

The model learns this equation:

```text
price = weight * size + bias
```

## Run

```bash
python "Numpy_linear /train.py"
```

## What To Interrogate

- What are `weight` and `bias`?
- What is a prediction?
- What is mean squared error?
- What are `d_weight` and `d_bias`?
- Why do we subtract the gradient?
- Why do we normalize `x` before training?

## The Core Training Loop

```python
predictions = weight * x_train_scaled + bias
errors = predictions - y_train

d_weight = (2 / n_samples) * np.sum(errors * x_train_scaled)
d_bias = (2 / n_samples) * np.sum(errors)

weight -= learning_rate * d_weight
bias -= learning_rate * d_bias
```

That is the small version of what machine learning training does.
