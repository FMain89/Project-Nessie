# Name: Freddie Main III
# GitHub: Fmain89
# Email:
# Description:

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import (
    Model, Layer_Dense, Activation_ReLU, Activation_Softmax,
    Loss_CategoricalCrossEntropy, Optimizer_Adam, Accuracy_Categorical
)

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Preprocess the data
train_images = np.concatenate([x.view(-1, 28*28).numpy() for x, _ in
                               train_loader], axis=0)
train_labels = np.concatenate([y.numpy() for _, y in train_loader], axis=0)
test_images = np.concatenate([x.view(-1, 28*28).numpy() for x, _ in
                              test_loader], axis=0)
test_labels = np.concatenate([y.numpy() for _, y in test_loader], axis=0)

# Initialize the model
model = Model()

# Add layers to the model
model.add(Layer_Dense(28*28, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

# Compile the model
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.1, decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Redirect output to a TXT file
with open('training_output.txt', 'w') as f:
    # Train the model
    model.train(
        train_images,
        train_labels,
        epochs=20,
        batch_size=64,
        print_every=10,
        validation_data=(test_images, test_labels)
    )
