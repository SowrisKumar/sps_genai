import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from app.cnn_model import SimpleCNN


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)


model = SimpleCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(2):
    model.train()

    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch + 1}, Loss: "
        f"{running_loss / len(trainloader):.4f}"
    )


torch.save(
    model.state_dict(),
    "cnn_model.pth"
)

print("Training complete")
