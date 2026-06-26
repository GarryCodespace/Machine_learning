# Tiny Transformer Language Model

Goal: learn the core mechanics behind GPT-style models with a small character-level transformer.

This project trains a model to predict the next character:

```text
input:  "the mode"
target: "he model"
```

Then it generates text one character at a time.

## Run Quick Smoke Test

```bash
python 12_tiny_transformer_language_model/train.py --max-steps 20 --eval-interval 10
```

Or with the local virtual environment:

```bash
.venv/bin/python 12_tiny_transformer_language_model/train.py --max-steps 20 --eval-interval 10
```

## Run Normal Training

```bash
python 12_tiny_transformer_language_model/train.py
```

## Run A Faster Small Model

```bash
python 12_tiny_transformer_language_model/train.py --max-steps 60 --eval-interval 30 --batch-size 8 --block-size 32 --embed-dim 32 --num-heads 4 --num-layers 2
```

## What This Teaches

- character tokenization
- vocabulary
- train/validation split
- context window
- token embeddings
- positional embeddings
- query/key/value
- causal self-attention
- transformer blocks
- next-token prediction
- `CrossEntropyLoss`
- text generation

## The Big Idea

A normal embeddings classifier does this:

```text
sentence -> embedding vectors -> label
```

A transformer language model does this:

```text
previous characters -> embedding vectors -> attention -> next character
```

The model is trained by shifting the text:

```text
input:  machine learnin
target: achine learning
```

Each position learns to predict the next character.

## Experiments

- Change `--max-steps`.
- Change `--block-size`.
- Change `--embed-dim`.
- Change `--num-layers`.
- Change `--num-heads`.
- Add more text to `data/tiny_stories.txt`.
- Try a different prompt with `--prompt`.
