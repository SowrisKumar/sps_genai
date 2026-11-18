import torch
import torchvision
import torchvision.transforms as transforms

from app.energy_model import EnergyModel


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
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


model = EnergyModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


def generate_samples(
    model,
    images,
    steps=20,
    step_size=10,
    noise_std=0.005
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


for epoch in range(2):

    model.train()

    running_loss = 0.0

    for real_images, _ in trainloader:

        real_images = real_images.to(device)

        batch_size = real_images.size(0)

        fake_images = torch.rand(
            batch_size,
            3,
            32,
            32,
            device=device
        ) * 2 - 1

        fake_images = generate_samples(
            model,
            fake_images
        )

        model.train()

        real_energy = model(real_images)
        fake_energy = model(fake_images)

        loss = (
            real_energy.mean()
            - fake_energy.mean()
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {running_loss / len(trainloader):.4f}"
    )


torch.save(
    model.state_dict(),
    "energy_model.pth"
)

print("Energy model training complete")
