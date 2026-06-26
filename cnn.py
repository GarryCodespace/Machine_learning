import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 input channels = RGB
        # 16 output channels = 16 kernels / detectors
        # detect edges, colors, textures, ...
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # 16 input feature maps -> 32 new feature maps
        # detect more complex patterns by combining the previous ones
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # downsample the feature maps by a factor of 2
        # this makes the model more efficient and also gives it some translation invariance
        self.pool = nn.MaxPool2d(2, 2)

        # CIFAR-10 image: 32x32
        # after pool: 32 -> 16 -> 8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        # 128 features -> 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: [batch, 3, 32, 32]
        # pool decreses dimensions by a factor of 2
        x = self.pool(F.relu(self.conv1(x)))
        # [batch, 16, 16, 16]

        x = self.pool(F.relu(self.conv2(x)))
        # [batch, 32, 8, 8]

        # flatten the feature maps into a single vector per image
        x = torch.flatten(x, 1)
        # [batch, 32*8*8]

        # fully connected layers for classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()

fake_image_batch = torch.randn(4, 3, 32, 32)
output = model(fake_image_batch)

print(output.shape)
# torch.Size([4, 10])