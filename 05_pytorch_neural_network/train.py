import torch
from torch import nn


class TinyXORNetwork(nn.Module):

# creates a network structure  when called tinuxornetwork
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super().__init__()

# build this neural networkrun this layer sequential (run the layers in order)
# linear represents connections and neurons 
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def main():
# set up random weights
    torch.manual_seed(1)

    X = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    model = TinyXORNetwork()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    epochs = 10_000

    for epoch in range(epochs):
        predictions = model(X)
        loss = loss_fn(predictions, y)
# fresh gradient each cycle
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:5d} | loss {loss.item():.4f}")

    with torch.no_grad():
        final_probabilities = model(X)
        final_classes = (final_probabilities >= 0.5).int()
        accuracy = (final_classes == y.int()).float().mean()

    print()
    print("Final predictions")
    for inputs, probability, predicted, target in zip(X, final_probabilities, final_classes, y):
        print(
            f"{inputs.int().tolist()} -> "
            f"probability={probability.item():.3f}, "
            f"predicted={predicted.item()}, "
            f"target={int(target.item())}"
        )

    print()
    print(f"Accuracy: {accuracy.item():.2f}")


if __name__ == "__main__":
    main()
