# Name: Freddie Main III
# GitHub: Fmain89
# Email:
# Description:

# import sys
import numpy as np
# import matplotlib

np.random.seed(0)


class Layer_Dense:
    """
    A fully connected (dense) layer of a neural network.

    This class represents a dense layer in a neural network, where each neuron
    in the layer is connected to every neuron in the previous layer. It
    includes methods to initialize the layer's weights and biases and to
    perform a forward  pass to compute the output of the layer.

    Attributes:
        weights (np.ndarray): The weight matrix of the layer, initialized with
                              small random values.
        biases (np.ndarray): The bias vector of the layer, initialized to
                             zeros.
        output (np.ndarray): The output of the layer after performing a forward
                             pass.
        dweights (np.ndarray): The gradient of the loss with respect to the
                               weights.
        dbiases (np.ndarray): The gradient of the loss with respect to the
                              biases.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs of the layer.
    """

    def __init__(self, num_inputs, num_neurons) -> None:
        """
        Initializes the Layer_Dense instance.

        This method initializes the weight matrix with small random values and
        the bias vector with zeros. The weights are scaled by 0.10 to keep the
        initial values small.

        Parameters:
            num_inputs (int): The number of input features (i.e., the number of
                              neurons in the previous layer).
            num_neurons (int): The number of neurons in the current layer.

        Returns:
            None
        """
        self.weights = 0.10 * np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, inputs) -> None:
        """
        Performs a forward pass through the layer.

        This method calculates the output of the dense layer by performing a
        dot product between the input data and the weight matrix, and then
        adding the bias vector.

        Parameters:
            inputs (np.ndarray): The input data or the output from the previous
                                 layer. It should have a shape of (n_samples,
                                 num_inputs).

        Returns:
            None: The output of the forward pass is stored in the `output`
                  attribute.
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the layer.

        This method calculates the gradients of the loss with respect to the
        layer's weights, biases, and inputs. These gradients are used during
        the optimization step to update the weights and biases.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the layer. It has the same shape as
                                  the `output` attribute.

        Returns:
            None: The gradients are stored in the `dweights`, `dbiases`, and
                  `dinputs` attributes.
        """

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    """
    Implements the ReLU (Rectified Linear Unit) activation function.

    The ReLU activation function is a non-linear activation function commonly
    used in neural networks. It effectively introduces non-linearity into the
    model while being computationally efficient. ReLU works by setting all
    negative input values to zero and leaving positive input values unchanged.

    Attributes:
        output (np.ndarray): The output of the ReLU activation function after
                             applying it to the input data.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs of the activation function.
    """

    def forward(self, inputs) -> None:
        """
        Performs a forward pass through the ReLU activation function.

        This method applies the ReLU activation function element-wise to the
        input data. Any negative values in the input are set to zero, and
        positive values are retained as is.

        Parameters:
            inputs (np.ndarray): The input data to the activation function. It
                                 should have a shape corresponding to the
                                 output of the previous layer (or the input
                                 data if this is the first layer).

        Returns:
            None: The output of the ReLU activation is stored in the `output`
                  attribute.
        """
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the ReLU activation function.

        This method calculates the gradient of the loss with respect to the
        inputs of the ReLU function during the backpropagation step. The
        gradient is only passed through the input values that were positive
        during the forward pass, as the gradient of ReLU is 0 for any input
        less than or equal to 0.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the activation function. It has the
                                  same shape as the `output` attribute.

        Returns:
            None: The gradients are stored in the `dinputs` attribute.
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    """
    Implements the Softmax activation function.

    The Softmax activation function is commonly used in the output layer of a
    neural network model for multi-class classification problems. It converts
    the logits (raw model outputs) into a probability distribution over the
    possible output classes, where the sum of the probabilities is 1.

    Attributes:
        output (np.ndarray): The output probabilities after applying the
                             Softmax function to the input data.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs of the activation function.
    """

    def forward(self, inputs) -> None:
        """
        Performs a forward pass through the Softmax activation function.

        This method applies the Softmax function to the input data,
        transforming the logits into probabilities. The Softmax
        function calculates the exponentials of the inputs, normalizes them by
        dividing by the sum of all exponentials in the batch, and ensures that
        the output values represent a probability distribution.

        Parameters:
            inputs (np.ndarray): The input data to the activation function,
                                 typically the logits from the last layer of
                                 the network. The shape should be (n_samples,
                                 n_classes), where n_samples is the number of
                                 samples and n_classes is the number of output
                                 classes.

        Returns:
            None: The output probabilities are stored in the `output`
                  attribute.
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the Softmax activation function.

        This method calculates the gradient of the loss with respect to the
        inputs of the Softmax function during the backpropagation step. The
        gradient is calculated using the Jacobian matrix of the Softmax
        function.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the activation function. It has the
                                  same shape as the `output` attribute.

        Returns:
            None: The gradients are stored in the `dinputs` attribute.
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(
                zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    """
    Base class for loss functions.

    This class serves as a base class for various loss functions used in
    neural networks. It provides a method to calculate the average loss over a
    batch of samples. The `forward` method, which computes the loss for each
    sample, should be implemented in subclasses.

    Attributes:
        None: The base `Loss` class does not define any attributes on its own.
    """

    def calculate(self, output, y):
        """
        Calculates the average loss over a batch of samples.

        This method computes the loss for each sample by calling the `forward`
        method (which must be implemented in a subclass), and then calculates
        the average loss across all samples in the batch.

        Parameters:
            output (np.ndarray): The predicted outputs from the neural network,
                                 typically the output of the final layer after
                                 applying an activation function. The shape
                                 should be (n_samples, n_classes) for
                                 classification tasks.
            y (np.ndarray): The true labels or target values. For
                            classification tasks, this could be an array of
                            class indices (shape: (n_samples,)) or one-hot
                            encoded labels (shape: (n_samples, n_classes)).

        Returns:
            float: The average loss over all samples in the batch.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy:
    """
    Implements the categorical cross-entropy loss function.

    This loss function is commonly used in classification tasks where the goal
    is to compare the predicted probabilities for each class with the actual
    class labels. It computes the negative log likelihood of the correct class,
    providing a measure of how well the model's predictions align with the true
    class labels.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the categorical cross-entropy loss.

        This method calculates the loss by computing the negative log
        likelihood for the true class labels. It supports both integer class
        labels and one-hot encoded labels.

        Parameters:
            y_pred (np.ndarray): The predicted probabilities from the model.
                                 The shape should be (n_samples, n_classes).
            y_true (np.ndarray): The true class labels. This can be an array of
                                 integer labels (shape: (n_samples,)) or a
                                 one-hot encoded array
                                 (shape: (n_samples, n_classes)).

        Returns:
            np.ndarray: An array of loss values for each sample in the batch.
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true) -> None:
        """
        Performs a backward pass through the categorical cross-entropy loss.

        This method calculates the gradient of the loss with respect to the
        inputs (the predicted probabilities). This is used during
        backpropagation to update the model's weights.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the previous layer (or activation
                                  function). The shape should be
                                  (n_samples, n_classes).
            y_true (np.ndarray): The true class labels. This can be an array of
                                 integer labels (shape: (n_samples,)) or a
                                 one-hot encoded array
                                 (shape: (n_samples, n_classes)).

        Returns:
            None: The gradients are stored in the `dinputs` attribute.
        """
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combines the Softmax activation function and the Categorical Cross-Entropy
    loss function into a single step.

    This combined class is useful for multi-class classification tasks, where
    the Softmax activation function is applied to the logits, followed by the
    computation of the categorical cross-entropy loss. This setup simplifies
    the gradient calculations during backpropagation.
    """

    def __init__(self) -> None:
        """
        Initializes the Activation_Softmax_Loss_CategoricalCrossentropy
        instance.

        This method initializes the necessary components, including the Softmax
        activation function and the Categorical Cross-Entropy loss function.

        Returns:
            None
        """
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true) -> float:
        """
        Performs a forward pass through the Softmax activation function and
        the categorical cross-entropy loss function.

        This method applies the Softmax activation function to the input logits
        and then calculates the categorical cross-entropy loss with respect to
        the true class labels.

        Parameters:
            inputs (np.ndarray): The input data to the activation function,
                                 typically the logits from the last layer of
                                 the network. The shape should be (n_samples,
                                 n_classes).
            y_true (np.ndarray): The true class labels. This can be an array of
                                 integer labels (shape: (n_samples,)) or a
                                 one-hot encoded array (shape: (n_samples,
                                 n_classes)).

        Returns:
            float: The average loss over the batch.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true) -> None:
        """
        Performs a backward pass through the combined Softmax activation
        function and categorical cross-entropy loss function.

        This method calculates the gradient of the loss with respect to the
        inputs, which is used during backpropagation to update the model's
        weights.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the previous layer. The shape
                                  should be (n_samples, n_classes).
            y_true (np.ndarray): The true class labels. This can be an array of
                                 integer labels (shape: (n_samples,)) or a
                                 one-hot encoded array (shape: (n_samples,
                                 n_classes)).

        Returns:
            None: The gradients are stored in the `dinputs` attribute.
        """
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# print("Python:", sys.version)
# print("Numpy:", np.__version__)
# print("Matplotlib:", matplotlib.__version__)
