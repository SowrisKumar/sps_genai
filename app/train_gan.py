import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from app.gan_model import Generator, Discriminator


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


trainset = torchvision.datasets.MNIST(
    root="./data_mnist",
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)


generator = Generator()
discriminator = Discriminator()


criterion = nn.BCELoss()

optimizer_g = optim.Adam(
    generator.parameters(),
    lr=0.0002
)

optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=0.0002
)


for epoch in range(2):
    for real_images, _ in trainloader:

        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)


        # Train Discriminator

        optimizer_d.zero_grad()

        real_output = discriminator(real_images)

        real_loss = criterion(
            real_output,
            real_labels
        )

        noise = torch.randn(batch_size, 100)

        fake_images = generator(noise)

        fake_output = discriminator(
            fake_images.detach()
        )

        fake_loss = criterion(
            fake_output,
            fake_labels
        )

        discriminator_loss = real_loss + fake_loss

        discriminator_loss.backward()

        optimizer_d.step()


        # Train Generator

        optimizer_g.zero_grad()

        fake_output = discriminator(fake_images)

        generator_loss = criterion(
            fake_output,
            real_labels
        )

        generator_loss.backward()

        optimizer_g.step()


    print(
        f"Epoch {epoch + 1}, "
        f"D Loss: {discriminator_loss.item():.4f}, "
        f"G Loss: {generator_loss.item():.4f}"
    )


torch.save(
    generator.state_dict(),
    "generator.pth"
)

torch.save(
    discriminator.state_dict(),
    "discriminator.pth"
)

print("Training complete")
