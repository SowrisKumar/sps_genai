import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            64 * 2 * 2,
            64
        )

        self.fc2 = nn.Linear(
            64,
            1
        )

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))

        x = self.flatten(x)

        x = swish(self.fc1(x))

        return self.fc2(x)
def generate_samples(
    model,
    images,
    steps=60,
    step_size=10,
    noise_std=0.01
):
    model.eval()

    for _ in range(steps):

        with torch.no_grad():
            noise = torch.randn_like(images) * noise_std
            images = (images + noise).clamp(-1, 1)

        images.requires_grad_(True)

        energy = model(images)

        gradients = torch.autograd.grad(
            energy,
            images,
            grad_outputs=torch.ones_like(energy)
        )[0]

        with torch.no_grad():
            gradients = gradients.clamp(-0.03, 0.03)

            images = (
                images - step_size * gradients
            ).clamp(-1, 1)

    return images.detach()
