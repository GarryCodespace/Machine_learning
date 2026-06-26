from __future__ import annotations

import argparse
import ssl
from urllib.error import URLError
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader 

from torchvision import datasets, transforms


DATA_DIR = Path(__file__).parent / "data"
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class CifarCnn(nn.Module):
    def __init__(self, channels=(32, 64, 128), num_classes=10):
        super().__init__()
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cifar10(train, transform):
    try:
        return datasets.CIFAR10(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=transform,
        )
    except (RuntimeError, URLError) as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error):
            raise

        print("CIFAR-10 download hit a local SSL certificate issue. Retrying download...")
        ssl._create_default_https_context = ssl._create_unverified_context
        return datasets.CIFAR10(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=transform,
        )


def make_data_loaders(batch_size, use_fake_data=False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if use_fake_data:
        train_dataset = datasets.FakeData(
            size=1024,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        test_dataset = datasets.FakeData(
            size=256,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
        )
    else:
        train_dataset = load_cifar10(train=True, transform=train_transform)
        test_dataset = load_cifar10(train=False, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, test_dataset, train_loader, test_loader


def count_correct(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).sum().item()


def run_epoch(model, loader, loss_fn, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += count_correct(logits, labels)
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def evaluate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += count_correct(logits, labels)
            total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def print_example_predictions(model, loader, device, limit=10):
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(images[:limit])
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1).values

    print()
    print("Example predictions")
    for actual, predicted, confidence in zip(labels[:limit], predictions, confidences):
        print(
            f"actual={CIFAR10_CLASSES[actual.item()]:10s} | "
            f"predicted={CIFAR10_CLASSES[predicted.item()]:10s} | "
            f"confidence={confidence.item():.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument(
        "--fake-data",
        action="store_true",
        help="Use generated CIFAR-shaped images for a fast code smoke test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)

    device = choose_device()
    train_dataset, test_dataset, train_loader, test_loader = make_data_loaders(
        batch_size=args.batch_size,
        use_fake_data=args.fake_data,
    )

    model = CifarCnn().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("CIFAR-10 CNN")
    print(f"Device:          {device}")
    print(f"Dataset:         {'FakeData smoke test' if args.fake_data else 'CIFAR-10'}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images:     {len(test_dataset)}")
    print(f"Classes:         {', '.join(CIFAR10_CLASSES)}")
    print()

    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            max_batches=args.max_test_batches,
        )

        print(
            f"epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_accuracy:.3f}"
        )

    print_example_predictions(model, test_loader, device)


if __name__ == "__main__":
    main()
