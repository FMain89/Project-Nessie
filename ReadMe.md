
# Neural Network from Scratch

This repository contains an implementation of a neural network built from scratch in Python. The code provides basic components necessary to create, train, and optimize a neural network, including activation functions, loss functions, optimizers, and regularization techniques.

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

This project is a simple yet comprehensive implementation of a neural network in Python, designed for educational purposes. It demonstrates how to build a neural network from the ground up, covering everything from defining layers and activation functions to implementing various optimization algorithms and regularization techniques.

## Features

- **Fully Connected Layers**: Implementation of dense (fully connected) layers.
- **Activation Functions**:
  - ReLU (Rectified Linear Unit)
  - Softmax
- **Loss Functions**:
  - Categorical Cross-Entropy
- **Optimizers**:
  - Stochastic Gradient Descent (SGD)
  - Adagrad
  - RMSprop
  - Adam
- **Regularization**:
  - L1 and L2 regularization on weights and biases
- **Dropout**:
  - Dropout layer for regularization during training
- **Backpropagation**: Implementation of backward passes for updating the model parameters.
- **Integration**: Combination of Softmax activation with Categorical Cross-Entropy loss for streamlined gradient calculations.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FMain89/Project-Nessie.git
   cd Project-Nessie
   ```

2. **Install dependencies**:
   Ensure you have Python installed, and install necessary packages using pip:
   ```bash
   pip install numpy
   ```

## Usage

You can run the neural network by executing the `main.py` file. It contains examples of how to define the network architecture, apply the activation functions, compute the loss, and optimize the parameters using the various optimizers provided.

```bash
python main.py
```

## File Structure

- `main.py`: The main file containing all class definitions and an example workflow.
- `README.md`: This file, providing information about the project.

## Examples

### Defining a Simple Neural Network

Below is an example of how to define a simple neural network using the components provided in this repository:

```python
import numpy as np
from main import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy, Optimizer_SGD, Layer_Dropout

# Example input data
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([0, 1, 1])

# Define layers
dense1 = Layer_Dense(2, 3, weight_regularizer_l2=0.01)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.5)
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dropout1.forward(activation1.output)
dense2.forward(dropout1.output)
activation2.forward(dense2.output)

# Compute loss
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print(f"Loss: {loss}")
```

### Optimizing the Model

You can also optimize the model parameters using one of the optimizers:

```python
optimizer = Optimizer_SGD(learning_rate=0.01, decay=1e-6, momentum=0.9)
optimizer.update_params(dense1)
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request to contribute. Ensure your code follows the Flake8 style guide.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
