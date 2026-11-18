import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from app.diffusion_model import (
    UNet,
    DiffusionModel
)


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", device)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


trainset = torchvision.datasets.CIFAR10(
    root="./data_cifar10",
    train=True,
    download=True,
    transform=transform
)


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)


network = UNet(
    image_size=64,
    num_channels=3
)

model = DiffusionModel(
    network
).to(device)


optimizer = torch.optim.AdamW(
    model.network.parameters(),
    lr=0.001,
    weight_decay=0.0001
)


loss_fn = nn.L1Loss()

mean = torch.zeros(3)
std = torch.zeros(3)

total_samples = 0


for images, _ in trainloader:

    batch_size = images.size(0)

    images_flat = images.view(
        batch_size,
        3,
        -1
    )

    mean += (
        images_flat.mean(
            dim=(0, 2)
        ) * batch_size
    )

    std += (
        images_flat.std(
            dim=(0, 2)
        ) * batch_size
    )

    total_samples += batch_size


mean /= total_samples
std /= total_samples


print("Mean:", mean)
print("Std:", std)


model.set_normalizer(
    mean.reshape(
        1,
        3,
        1,
        1
    ).to(device),

    std.reshape(
        1,
        3,
        1,
        1
    ).to(device)
)


EPOCHS = 2


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, _ in trainloader:

        images = images.to(device)

        loss = model.train_step(
            images,
            optimizer,
            loss_fn
        )

        running_loss += loss

    average_loss = (
        running_loss
        / len(trainloader)
    )

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {average_loss:.4f}"
    )


torch.save(
    {
        "network": model.network.state_dict(),

        "ema_network":
            model.ema_network.state_dict(),

        "mean":
            model.normalizer_mean,

        "std":
            model.normalizer_std
    },

    "diffusion_model.pth"
)


print(
    "Diffusion model training complete"
)


