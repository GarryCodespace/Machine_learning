# MNIST Digit Classification

Goal: train a PyTorch neural network to classify handwritten digits from `0` to `9`.

This is the next step after XOR because MNIST introduces the normal deep learning workflow:

```text
dataset -> DataLoader -> batches -> model -> loss -> optimizer -> train -> evaluate
```

## Install

This project needs `torchvision` so PyTorch can download MNIST:

```bash
pip install -r requirements.txt
```

## Run

```bash
python 08_mnist_digit_classification/train.py
```

The first run downloads MNIST into:

```text
08_mnist_digit_classification/data/
```

## What To Learn

- What an image tensor is
- Why MNIST images have shape `1 x 28 x 28`
- What a batch is
- What `DataLoader` does
- Why the model outputs 10 numbers
- What `CrossEntropyLoss` expects
- Why we use logits instead of sigmoid for multi-class classification
- How training accuracy differs from test accuracy

## Things To Modify

- Change `hidden_size` from `128` to `32` or `256`.
- Change `learning_rate`.
- Change `epochs`.
- Add another hidden layer.
- Print wrong predictions and inspect what the model missed.
- Replace the simple neural network with a CNN.
