# Embeddings Text Classifier

Goal: learn how text becomes numbers that a neural network can train on.

This project predicts review sentiment:

```text
sentence -> positive or negative
```

It uses a small offline dataset and a PyTorch model with `nn.Embedding`.

## Run

```bash
python 11_embeddings_text_classifier/train.py
```

Or with the local virtual environment:

```bash
.venv/bin/python 11_embeddings_text_classifier/train.py
```

## What This Teaches

- tokenization
- vocabulary building
- integer encoding
- padding variable-length sentences
- `nn.Embedding`
- mean pooling
- binary text classification
- `BCEWithLogitsLoss`
- why embeddings are the bridge toward transformers

## Mental Model

Raw text cannot go directly into a neural network:

```text
"this movie was fantastic" -> ?
```

So we convert it:

```text
tokens -> token ids -> embedding vectors -> sentence vector -> prediction
```

Example:

```text
"this movie was fantastic"
["this", "movie", "was", "fantastic"]
[2, 15, 8, 31]
```

`nn.Embedding` learns a vector for each token ID. During training, useful words move toward useful vector positions.

## Experiments

- Change `embedding_dim`.
- Change `hidden_size`.
- Add more reviews to the CSV.
- Print the vocabulary.
- Compare average embeddings vs using only the first token.
- Try your own positive and negative sentences.

## Note On Accuracy

This dataset is intentionally tiny so you can understand every moving part.
Because it has only 50 rows, the model can overfit quickly:

```text
train accuracy gets very high
test accuracy may bounce around
```

That is normal for a tiny learning dataset. The main goal here is understanding embeddings, padding, and the text training loop.
