import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(100, 7 * 7 * 128)

        self.deconv1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(
            64,
            1,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)

        x = x.view(-1, 128, 7, 7)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(128)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc = nn.Linear(
            128 * 7 * 7,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.sigmoid(x)

        return x
