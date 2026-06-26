# CIFAR-10 CNN

Goal: train a convolutional neural network on CIFAR-10 with PyTorch.

CIFAR-10 is a classic image classification dataset with 10 classes:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

Unlike MNIST, CIFAR-10 images are color images:

```text
3 channels x 32 height x 32 width
```

## Run A Quick Smoke Test

First use the project virtual environment:

```bash
cd ~/machine_learning
source .venv/bin/activate
```

This checks the model and training loop instantly with generated CIFAR-shaped images:

```bash
python 10_cifar10_cnn/train.py --fake-data --epochs 1 --max-train-batches 5 --max-test-batches 2
```

This checks the real CIFAR-10 data path without training for long:

```bash
python 10_cifar10_cnn/train.py --epochs 1 --max-train-batches 5 --max-test-batches 2
```

## Run Normal Training

```bash
python 10_cifar10_cnn/train.py --epochs 5
```

For a better result, try:

```bash
python 10_cifar10_cnn/train.py --epochs 15
```

If you are already inside `10_cifar10_cnn`, run the script with the parent repo venv:

```bash
../.venv/bin/python train.py --fake-data --epochs 1 --max-train-batches 5 --max-test-batches 2
```

## What This Teaches

- `torchvision.datasets.CIFAR10`
- train vs test transforms
- data augmentation
- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d`
- flattening CNN features
- `CrossEntropyLoss`
- model evaluation
- experimenting with architecture

## Important Ideas

### Data Augmentation

Training images are randomly changed a little:

```python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip()
```

This helps the model learn more general image patterns.

### CNN Blocks

The model uses blocks like:

```text
Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
```

The early layers learn simple patterns like edges and colors. Later layers learn more useful image features.

## Experiments

- Change `epochs`.
- Change `learning_rate`.
- Remove data augmentation and compare accuracy.
- Add another convolution block.
- Change `channels` from `(32, 64, 128)` to `(64, 128, 256)`.
- Try `Dropout`.
- Print incorrect predictions and inspect what the model confuses.
