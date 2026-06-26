# Machine Learning

Beginner ML projects for learning by running, inspecting, modifying, and rebuilding.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Projects

Run the Titanic survival classifier:

```bash
python 01_titanic_survival/train.py
```

Run the house price regression model:

```bash
python 02_house_price_regression/train.py
```

Run the spam detector:

```bash
python 03_spam_detector/train.py
```

Run the tiny neural network from scratch:

```bash
python 04_tiny_neural_network/train.py
```

Run the PyTorch neural network:

```bash
python 05_pytorch_neural_network/train.py
```

Run NumPy linear regression from scratch:

```bash
python 06_linear_regression_from_scratch/train.py
```

Run NumPy logistic regression from scratch:

```bash
python 07_logistic_regression_from_scratch/train.py
```

Run the MNIST digit classifier:

```bash
python 08_mnist_digit_classification/train.py
```

Open the EDA notebook:

```text
09_data_processing_and_eda/eda_titanic.ipynb
```

Run the CIFAR-10 CNN:

```bash
python 10_cifar10_cnn/train.py --fake-data --epochs 1 --max-train-batches 5 --max-test-batches 2
```

Run the embeddings text classifier:

```bash
python 11_embeddings_text_classifier/train.py
```

Run the tiny transformer language model:

```bash
python 12_tiny_transformer_language_model/train.py --max-steps 20 --eval-interval 10
```

## Study Loop

1. Run one project.
2. Ask AI to explain the code section by section.
3. Add print statements to inspect the data.
4. Change one thing and rerun it.
5. Rewrite the skeleton in a new file from memory.
