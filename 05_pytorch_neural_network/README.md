# PyTorch Neural Network

Goal: rebuild the XOR neural network using PyTorch.

The previous NumPy project showed the training math by hand. This version uses PyTorch for automatic gradients:

```text
loss.backward()
optimizer.step()
```

The model still learns XOR:

```text
0, 0 -> 0
0, 1 -> 1
1, 0 -> 1
1, 1 -> 0
```

## Run

```bash
python 05_pytorch_neural_network/train.py
```

## What To Interrogate

- What is a tensor?
- What is `nn.Module`?
- What does `forward()` do?
- What does `loss.backward()` calculate?
- Why do we call `optimizer.zero_grad()`?
- What does `optimizer.step()` change?
- How is this different from the NumPy version?

## Modifications

- Change `hidden_size` from `4` to `2` or `8`.
- Change the learning rate.
- Change `nn.Tanh()` to `nn.ReLU()`.
- Replace `torch.optim.SGD` with `torch.optim.Adam`.
- Print model weights before and after training.
