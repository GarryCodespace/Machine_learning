# Tiny Neural Network From Scratch

Goal: learn what a neural network does under `.fit()` by building one with only NumPy.

This project trains a tiny neural network to solve XOR:

```text
0, 0 -> 0
0, 1 -> 1
1, 0 -> 1
1, 1 -> 0
```

XOR is useful because a straight line cannot separate the classes. A neural network with a hidden layer can.

## Run

```bash
python 04_tiny_neural_network/train.py
```

## What To Interrogate

- What are weights and biases?
- What is a hidden layer?
- Why does this use `tanh` in the hidden layer?
- Why does this use `sigmoid` at the output?
- What is binary cross-entropy loss?
- What happens during backpropagation?
- What does the learning rate control?

## Modifications

- Change `hidden_size` from `4` to `2` or `8`.
- Change `learning_rate`.
- Change the number of `epochs`.
- Print the hidden layer values.
- Replace `tanh` with ReLU.
- Add more noisy training points around the XOR corners.
