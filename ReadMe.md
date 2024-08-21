
# Neural Network from Scratch

This repository contains an implementation of a neural network built from scratch in Python. The code provides all the essential components necessary to create, train, and optimize a neural network, including activation functions, loss functions, optimizers, and regularization techniques.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates how to build a neural network from scratch without relying on external machine learning libraries such as TensorFlow or PyTorch. The implementation includes basic layers, various activation functions, optimizers, and loss functions, enabling users to experiment and understand the underlying mechanics of neural networks.

## Features

- **Layers**: Dense layers, Dropout layers, Input layer.
- **Activation Functions**: ReLU, Softmax, Sigmoid, Linear.
- **Loss Functions**: Categorical Cross-Entropy, Binary Cross-Entropy, Mean Squared Error, Mean Absolute Error.
- **Optimizers**: Stochastic Gradient Descent (SGD), Adagrad, RMSprop, Adam.
- **Regularization**: L1 and L2 regularization for weights and biases.
- **Accuracy Metrics**: Support for categorical and regression accuracy calculations.
- **Combined Functions**: Softmax activation combined with categorical cross-entropy loss for efficient backpropagation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fmain89/Project-Nessie.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Project-Nessie
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Example: Training a Simple Neural Network

```python
import numpy as np
from main import Model, Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy, Optimizer_Adam, Accuracy_Categorical

# Sample data
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 2])

# Initialize the model
model = Model()
model.add(Layer_Dense(3, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 3))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=100, print_every=10)
```

## File Structure

- `main.py`: Contains the implementation of all layers, activation functions, loss functions, optimizers, and the model.
- `ReadMe.md`: Documentation for the project.

## Examples

### Training with Different Loss Functions and Optimizers

You can easily switch between different loss functions and optimizers by modifying the corresponding lines in the code. For example:

```python
model.set(
    loss=Loss_MeanSquareError(),
    optimizer=Optimizer_RMSprop(),
    accuracy=Accuracy_Regresson()
)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
