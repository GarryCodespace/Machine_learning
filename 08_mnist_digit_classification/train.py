from pathlib import Path
import ssl

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).parent / "data"


class DigitClassifier(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.network(x)


def count_correct_from_logits(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).sum().item()


def load_mnist(train, transform):
    try:
        return datasets.MNIST(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=transform,
        )
    except RuntimeError as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error):
            raise

        print("MNIST download hit a local SSL certificate issue. Retrying download...")
        ssl._create_default_https_context = ssl._create_unverified_context
        return datasets.MNIST(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=transform,
        )


def main():
    torch.manual_seed(42)

    transform = transforms.ToTensor()

    train_dataset = load_mnist(train=True, transform=transform)
    test_dataset = load_mnist(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = DigitClassifier(hidden_size=128)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    print("MNIST digit classifier")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images:     {len(test_dataset)}")
    print()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for images, labels in train_loader:
            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += count_correct_from_logits(logits, labels)
            total_seen += batch_size

        average_loss = total_loss / total_seen
        average_accuracy = total_correct / total_seen

        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"train loss {average_loss:.4f} | "
            f"train accuracy {average_accuracy:.3f}"
        )

    model.eval()
    test_correct = 0
    test_seen = 0

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            test_correct += count_correct_from_logits(logits, labels)
            test_seen += labels.size(0)

    test_accuracy = test_correct / test_seen

    print()
    print(f"Test accuracy: {test_accuracy:.3f}")
    print()
    print("Example predictions")

    images, labels = next(iter(test_loader))
    with torch.no_grad():
        logits = model(images[:10])
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    for actual, predicted, confidence in zip(
        labels[:10],
        predictions,
        probabilities.max(dim=1).values,
    ):
        print(
            f"actual={actual.item()} | "
            f"predicted={predicted.item()} | "
            f"confidence={confidence.item():.3f}"
        )


if __name__ == "__main__":
    main()
