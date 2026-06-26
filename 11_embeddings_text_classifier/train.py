from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


DATA_PATH = Path(__file__).parent / "data" / "reviews_tiny.csv"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def load_rows(path):
    with path.open(newline="") as file:
        return [
            {"text": row["text"], "label": int(row["label"])}
            for row in csv.DictReader(file)
        ]


def build_vocab(rows, min_count=1):
    counts = Counter()
    for row in rows:
        counts.update(tokenize(row["text"]))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counts.items()):
        if count >= min_count:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]


class ReviewDataset(Dataset):
    def __init__(self, rows, vocab):
        self.examples = [
            (
                torch.tensor(encode_text(row["text"], vocab), dtype=torch.long),
                torch.tensor(row["label"], dtype=torch.float32),
            )
            for row in rows
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch):
    token_ids, labels = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long)
    max_length = lengths.max().item()

    padded = torch.zeros((len(token_ids), max_length), dtype=torch.long)
    for row_index, ids in enumerate(token_ids):
        padded[row_index, : len(ids)] = ids

    labels = torch.stack(labels).unsqueeze(1)
    return padded, lengths, labels


class EmbeddingSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, token_ids, lengths):
        embeddings = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        averaged = summed / lengths.unsqueeze(1).clamp(min=1)
        return self.classifier(averaged)


def accuracy_from_logits(logits, labels):
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    return (predictions == labels).float().mean().item()


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    with torch.no_grad():
        for token_ids, lengths, labels in loader:
            logits = model(token_ids, lengths)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_accuracy += accuracy_from_logits(logits, labels)
            total_batches += 1

    return total_loss / total_batches, total_accuracy / total_batches


def predict_examples(model, vocab, examples):
    model.eval()
    encoded = [
        (
            torch.tensor(encode_text(text, vocab), dtype=torch.long),
            torch.tensor(0.0),
        )
        for text in examples
    ]
    token_ids, lengths, _ = collate_batch(encoded)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(token_ids, lengths)).squeeze(1)

    print()
    print("Example predictions")
    for text, probability in zip(examples, probabilities):
        label = "positive" if probability.item() >= 0.5 else "negative"
        print(f"{text!r} -> {label} ({probability.item():.3f})")


def main():
    torch.manual_seed(42)

    rows = load_rows(DATA_PATH)
    vocab = build_vocab(rows)
    dataset = ReviewDataset(rows, vocab)

    generator = torch.Generator().manual_seed(42)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = EmbeddingSentimentClassifier(vocab_size=len(vocab))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 30

    print("Embeddings text classifier")
    print(f"Rows:        {len(rows)}")
    print(f"Vocabulary:  {len(vocab)} tokens")
    print(f"Train rows:  {len(train_dataset)}")
    print(f"Test rows:   {len(test_dataset)}")
    print()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0

        for token_ids, lengths, labels in train_loader:
            logits = model(token_ids, lengths)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy_from_logits(logits, labels)
            total_batches += 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            train_loss = total_loss / total_batches
            train_accuracy = total_accuracy / total_batches
            test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)
            print(
                f"epoch {epoch + 1:2d}/{epochs} | "
                f"train loss {train_loss:.4f} | "
                f"train acc {train_accuracy:.3f} | "
                f"test loss {test_loss:.4f} | "
                f"test acc {test_accuracy:.3f}"
            )

    print()
    print("Vocabulary sample")
    for token, token_id in list(vocab.items())[:20]:
        print(f"{token_id:2d}: {token}")

    predict_examples(
        model,
        vocab,
        [
            "this movie was fun and beautiful",
            "the product is broken and awful",
            "the lesson was clear and helpful",
            "the app feels slow and confusing",
        ],
    )


if __name__ == "__main__":
    main()
