# Name: Freddie Main III
# GitHub: Fmain89
# Email:
# Description:

# import sys
import numpy as np
# import matplotlib


class Layer_Dense:
    """
    A fully connected (dense) layer of a neural network.

    This class represents a dense layer in a neural network, where each neuron
    in the layer is connected to every neuron in the previous layer. It
    includes methods to initialize the layer's weights and biases and to
    perform a forward pass to compute the output of the layer.

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
        weight_regularizer_l1 (float): L1 regularization factor for the
                                       weights.
        weight_regularizer_l2 (float): L2 regularization factor for the
                                       weights.
        bias_regularizer_l1 (float): L1 regularization factor for the biases.
        bias_regularizer_l2 (float): L2 regularization factor for the biases.
    """

    def __init__(self, num_inputs, num_neurons, weight_regularizer_l1=0,
                 weight_regularizer_l2=0, bias_regularizer_l1=0,
                 bias_regularizer_l2=0) -> None:
        """
        Initializes the Layer_Dense instance.

        This method initializes the weight matrix with small random values and
        the bias vector with zeros. The weights are scaled by 0.10 to keep the
        initial values small.

        Parameters:
            num_inputs (int): The number of input features (i.e., the number of
                              neurons in the previous layer).
            num_neurons (int): The number of neurons in the current layer.
            weight_regularizer_l1 (float): L1 regularization factor for the
                                           weights.
            weight_regularizer_l2 (float): L2 regularization factor for the
                                           weights.
            bias_regularizer_l1 (float): L1 regularization factor for the
                                         biases.
            bias_regularizer_l2 (float): L2 regularization factor for the
                                         biases.

        Returns:
            None
        """
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

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

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout:
    """
    Implements a dropout layer.

    Dropout is a regularization technique that randomly sets a fraction of the
    input units to 0 at each update during training time, which helps prevent
    overfitting. The layer temporarily drops out (deactivates) a proportion of
    the input units during the forward pass.

    Attributes:
        rate (float): The dropout rate (fraction of input units to drop). The
                      value should be between 0 and 1.
        binary_mask (np.ndarray): The binary mask applied to the inputs,
                                  indicating which inputs were kept and which
                                  were dropped.
        inputs (np.ndarray): The input values to the layer during the forward
                             pass.
        output (np.ndarray): The output values after applying the dropout mask.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs, after applying the dropout mask during
                              backpropagation.
    """

    def __init__(self, rate) -> None:
        """
        Initializes the Layer_Dropout instance.

        This method sets the dropout rate, which determines the fraction of
        input units to drop (deactivate) during the forward pass.

        Parameters:
            rate (float): The dropout rate, a float between 0 and 1. This
                          specifies the fraction of input units to drop.

        Returns:
            None
        """
        self.rate = 1 - rate

    def forward(self, inputs) -> None:
        """
        Performs a forward pass through the dropout layer.

        During the forward pass, a binary mask is created using a binomial
        distribution based on the dropout rate. The input values are then
        multiplied by this mask to drop (deactivate) a portion of the inputs.

        Parameters:
            inputs (np.ndarray): The input values to the layer. The shape
                                 should match the output of the previous layer.

        Returns:
            None: The output after applying dropout is stored in the `output`
                  attribute.
        """
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the dropout layer.

        During backpropagation, the gradient of the loss with respect to the
        inputs is calculated by multiplying the incoming gradient by the same
        binary mask used during the forward pass. This ensures that the
        gradient is only propagated through the units that were not dropped.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the dropout layer.

        Returns:
            None: The gradients after applying the dropout mask are stored in
                  the `dinputs` attribute.
        """
        self.dinputs = dvalues * self.binary_mask


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
        self.inputs = inputs
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

    def regularization_loss(self, layer):
        """
        Calculates the regularization loss for a given layer.

        This method computes the additional loss introduced by L1 and L2
        regularization on the weights and biases of a layer. The regularization
        loss is a penalty that discourages complex models by adding a cost to
        larger weight values.

        Parameters:
            layer: The layer whose regularization loss is to be calculated. The
                    layer should have attributes `weights`, `biases`,
                    `weight_regularizer_l1`, `weight_regularizer_l2`,
                    `bias_regularizer_l1`, and `bias_regularizer_l2`.

        Returns:
            float: The calculated regularization loss.
        """
        regularization_loss = 0
        if layer.weight_regularization_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(
                layer.weights))

        if layer.weight_regularization_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(
                layer.weights * layer.weights)

        if layer.bias_regularization_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(
                    layer.bias))

        if layer.bias_regularization_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(
                layer.bias * layer.bias)

        return regularization_loss

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


class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer with optional learning rate
    decay and momentum.

    This optimizer adjusts the weights and biases of the layers in a neural
    network based on the gradients calculated during backpropagation. It
    includes options for learning rate decay, which reduces the learning rate
    over time, and momentum, which helps accelerate gradient vectors in the
    right directions, leading to faster converging.

    Attributes:
        learning_rate (float): The initial learning rate for the optimizer.
        current_learning_rate (float): The adjusted learning rate considering
                                       decay.
        decay (float): The decay factor to decrease the learning rate over
                       iterations.
        iterations (int): The number of iterations (updates) performed.
        momentum (float): The momentum factor to accelerate gradient descent.
    """

    def __init__(self, learning_rate=1.0, decay=0, momentum=0) -> None:
        """
        Initializes the Optimizer_SGD instance.

        This method initializes the optimizer with the given learning rate,
        decay, and momentum. It also sets up the initial learning rate and
        iteration count.

        Parameters:
            learning_rate (float): The initial learning rate (default is 1.0).
            decay (float): The decay factor to decrease the learning rate over
                           time (default is 0, meaning no decay).
            momentum (float): The momentum factor to use (default is 0,
                              meaning no momentum).

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self) -> None:
        """
        Adjusts the learning rate before updating parameters.

        If decay is used, this method adjusts the learning rate based on the
        number of iterations, reducing it over time to allow for finer
        adjustments to the model parameters as training progresses.

        Returns:
            None
        """
        if self.decay:
            self.current_learning_rate = (
                self.learning_rate * (1. / (1. + self.decay * self.iterations))
            )

    def update_params(self, layer) -> None:
        """
        Updates the weights and biases of a layer.

        This method applies the calculated gradients to the weights and biases
        of the given layer. If momentum is used, it applies momentum to the
        updates to accelerate gradient descent.

        Parameters:
            layer: The layer whose parameters (weights and biases) will be
                   updated. The layer should have attributes `weights`,
                   `biases`, `dweights`, and `dbiases`.

        Returns:
            None
        """
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = (
                self.momentum * layer.weight_momentums -
                self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates

            bias_updates = (
                self.momentum * layer.bias_momentums -
                self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self) -> None:
        """
        Increments the iteration count after parameter updates.

        This method is called after the parameters of all layers have been
        updated. It increments the internal counter for the number of
        iterations, which can be used to adjust the learning rate in
        subsequent updates.

        Returns:
            None
        """
        self.iterations += 1


class Optimizer_Adagrad:
    """
    Implements the Adagrad optimizer.

    The Adagrad (Adaptive Gradient) optimizer adapts the learning rate for each
    parameter individually by scaling it inversely proportional to the square
    root of the sum of the squares of all historical gradients. This allows
    for more fine-tuned updates and can be particularly effective for sparse
    data.

    Attributes:
        learning_rate (float): The initial learning rate for the optimizer.
        current_learning_rate (float): The adjusted learning rate considering
                                       decay.
        decay (float): The decay factor to decrease the learning rate over
                       iterations.
        iterations (int): The number of iterations (updates) performed.
        epsilon (float): A small constant to prevent division by zero.
    """

    def __init__(self, learning_rate=1.0, decay=0, epsilon=1e-7) -> None:
        """
        Initializes the Optimizer_Adagrad instance.

        This method initializes the optimizer with the given learning rate,
        decay, and epsilon. It also sets up the initial learning rate,
        iteration count, and caches for storing squared gradients.

        Parameters:
            learning_rate (float): The initial learning rate (default is 1.0).
            decay (float): The decay factor to decrease the learning rate over
                           time (default is 0, meaning no decay).
            epsilon (float): A small constant added to the denominator to
                             prevent division by zero (default is 1e-7).

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self) -> None:
        """
        Adjusts the learning rate before updating parameters.

        If decay is used, this method adjusts the learning rate based on the
        number of iterations, reducing it over time to allow for finer
        adjustments to the model parameters as training progresses.

        Returns:
            None
        """
        if self.decay:
            self.current_learning_rate = (
                self.learning_rate * (1. / (1. + self.decay * self.iterations))
            )

    def update_params(self, layer) -> None:
        """
        Updates the weights and biases of a layer using Adagrad optimization.

        This method applies the Adagrad update rule, which adapts the learning
        rate based on the accumulated squared gradients for each parameter.
        It ensures that the learning rate decreases more slowly for parameters
        with larger gradients.

        Parameters:
            layer: The layer whose parameters (weights and biases) will be
                   updated. The layer should have attributes `weights`,
                   `biases`, `dweights`, and `dbiases`.

        Returns:
            None
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (
            np.sqrt(layer.weight_cache) + self.epsilon
        )
        layer.biases += -self.current_learning_rate * layer.dbiases / (
            np.sqrt(layer.bias_cache) + self.epsilon
        )

    def post_update_params(self) -> None:
        """
        Increments the iteration count after parameter updates.

        This method is called after the parameters of all layers have been
        updated. It increments the internal counter for the number of
        iterations, which can be used to adjust the learning rate in
        subsequent updates.

        Returns:
            None
        """
        self.iterations += 1


class Optimizer_RMSprop:
    """
    Implements the RMSprop optimizer.

    RMSprop (Root Mean Square Propagation) is an adaptive learning rate method
    that adjusts the learning rate for each parameter individually. It keeps a
    moving average of the squared gradients and divides the gradient by the
    square root of this average. This helps to normalize the updates and can
    lead to better performance in training deep neural networks.

    Attributes:
        learning_rate (float): The initial learning rate for the optimizer.
        current_learning_rate (float): The adjusted learning rate considering
                                       decay.
        decay (float): The decay factor to decrease the learning rate over
                       iterations.
        iterations (int): The number of iterations (updates) performed.
        epsilon (float): A small constant to prevent division by zero.
        rho (float): The decay rate for the moving average of squared
                     gradients.
    """

    def __init__(self, learning_rate=0.001, decay=0,
                 epsilon=1e-7, rho=0.9) -> None:
        """
        Initializes the Optimizer_RMSprop instance.

        This method initializes the optimizer with the given learning rate,
        decay, epsilon, and rho (decay rate for the moving average). It also
        sets up the initial learning rate and iteration count.

        Parameters:
            learning_rate (float): The initial learning rate (default is 0.001)
            decay (float): The decay factor to decrease the learning rate over
                           time (default is 0, meaning no decay).
            epsilon (float): A small constant added to the denominator to
                             prevent division by zero (default is 1e-7).
            rho (float): The decay rate for the moving average of squared
                         gradients (default is 0.9).

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self) -> None:
        """
        Adjusts the learning rate before updating parameters.

        If decay is used, this method adjusts the learning rate based on the
        number of iterations, reducing it over time to allow for finer
        adjustments to the model parameters as training progresses.

        Returns:
            None
        """
        if self.decay:
            self.current_learning_rate = (
                self.learning_rate * (1. / (1. + self.decay * self.iterations))
            )

    def update_params(self, layer) -> None:
        """
        Updates the weights and biases of a layer using RMSprop optimization.

        This method applies the RMSprop update rule, which keeps a moving
        average of the squared gradients and normalizes the updates by
        dividing by the square root of this average.

        Parameters:
            layer: The layer whose parameters (weights and biases) will be
                   updated. The layer should have attributes `weights`,
                   `biases`, `dweights`, and `dbiases`.

        Returns:
            None
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (
            self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        )
        layer.bias_cache = (
            self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        )

        layer.weights += -self.current_learning_rate * layer.dweights / (
            np.sqrt(layer.weight_cache) + self.epsilon
        )
        layer.biases += -self.current_learning_rate * layer.dbiases / (
            np.sqrt(layer.bias_cache) + self.epsilon
        )

    def post_update_params(self) -> None:
        """
        Increments the iteration count after parameter updates.

        This method is called after the parameters of all layers have been
        updated. It increments the internal counter for the number of
        iterations, which can be used to adjust the learning rate in
        subsequent updates.

        Returns:
            None
        """
        self.iterations += 1


class Optimizer_Adam:
    """
    Implements the Adam optimizer.

    Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization
    algorithm that's been designed to combine the advantages of two other
    popular methods: AdaGrad and RMSProp. It computes adaptive learning rates
    for each parameter by keeping track of the first and second moments of the
    gradients.

    Attributes:
        learning_rate (float): The initial learning rate for the optimizer.
        current_learning_rate (float): The adjusted learning rate considering
                                       decay.
        decay (float): The decay factor to decrease the learning rate over
                       iterations.
        iterations (int): The number of iterations (updates) performed.
        epsilon (float): A small constant to prevent division by zero.
        beta_1 (float): The exponential decay rate for the first moment
                        estimates.
        beta_2 (float): The exponential decay rate for the second moment
                        estimates.
    """

    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999) -> None:
        """
        Initializes the Optimizer_Adam instance.

        This method initializes the optimizer with the given learning rate,
        decay, epsilon, beta_1, and beta_2. It also sets up the initial
        learning rate and iteration count.

        Parameters:
            learning_rate (float): The initial learning rate (default is 0.001)
            decay (float): The decay factor to decrease the learning rate over
                           time (default is 0, meaning no decay).
            epsilon (float): A small constant added to the denominator to
                             prevent division by zero (default is 1e-7).
            beta_1 (float): The exponential decay rate for the first moment
                            estimates (default is 0.9).
            beta_2 (float): The exponential decay rate for the second moment
                            estimates (default is 0.999).

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self) -> None:
        """
        Adjusts the learning rate before updating parameters.

        If decay is used, this method adjusts the learning rate based on the
        number of iterations, reducing it over time to allow for finer
        adjustments to the model parameters as training progresses.

        Returns:
            None
        """
        if self.decay:
            self.current_learning_rate = (
                self.learning_rate * (1. / (1. + self.decay * self.iterations))
            )

    def update_params(self, layer) -> None:
        """
        Updates the weights and biases of a layer using Adam optimization.

        This method applies the Adam update rule, which uses the first and
        second moment estimates to adapt the learning rate for each parameter
        individually.

        Parameters:
            layer: The layer whose parameters (weights and biases) will be
                   updated. The layer should have attributes `weights`,
                   `biases`, `dweights`, and `dbiases`.

        Returns:
            None
        """
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) *
            layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) *
            layer.dbiases
        )

        weight_momentums_corrected = (
            layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        )
        bias_momentums_corrected = (
            layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        )

        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) *
            layer.dweights ** 2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) *
            layer.dbiases ** 2
        )

        weight_cache_corrected = (
            layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        )
        bias_cache_corrected = (
            layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        )

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
            np.sqrt(weight_cache_corrected) + self.epsilon
        )
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
            np.sqrt(bias_cache_corrected) + self.epsilon
        )

    def post_update_params(self) -> None:
        """
        Increments the iteration count after parameter updates.

        This method is called after the parameters of all layers have been
        updated. It increments the internal counter for the number of
        iterations, which can be used to adjust the learning rate in
        subsequent updates.

        Returns:
            None
        """
        self.iterations += 1


# print("Python:", sys.version)
# print("Numpy:", np.__version__)
# print("Matplotlib:", matplotlib.__version__)
