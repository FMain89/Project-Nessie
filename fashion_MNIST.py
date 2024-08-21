# Name: Freddie Main III
# GitHub: Fmain89
# Email:
# Description:

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import (
    Model, Layer_Dense, Activation_ReLU, Activation_Softmax,
    Loss_CategoricalCrossEntropy, Optimizer_Adam, Accuracy_Categorical
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


train_images = torch.cat(
    [x.view(-1, 28*28) for x, _ in train_loader], dim=0).cpu().numpy()
train_labels = torch.cat([y for _, y in train_loader], dim=0).cpu().numpy()
test_images = torch.cat(
    [x.view(-1, 28*28) for x, _ in test_loader], dim=0).cpu().numpy()
test_labels = torch.cat([y for _, y in test_loader], dim=0).cpu().numpy()


model = Model()

model.add(Layer_Dense(28*28, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.finalize()


model.train(
    train_images,
    train_labels,
    epochs=20,
    batch_size=64,
    print_every=10,
    validation_data=(test_images, test_labels)
)
