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
    includes methods to initialize the layer's weights and biases, perform a
    forward pass to compute the output of the layer, and calculate gradients
    during backpropagation for the purpose of optimization.

    Attributes:
        weights (np.ndarray): The weight matrix of the layer, initialized with
                              small random values, scaled by 0.01.
        biases (np.ndarray): The bias vector of the layer, initialized to
                             zeros.
        output (np.ndarray): The output of the layer after performing a forward
                             pass.
        dweights (np.ndarray): The gradient of the loss with respect to the
                               weights, used during backpropagation.
        dbiases (np.ndarray): The gradient of the loss with respect to the
                              biases, used during backpropagation.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs of the layer, used during backpropagation.
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

        This method initializes the weight matrix with small random values
        (scaled by 0.01) and the bias vector with zeros. Additionally, it sets
        up regularization factors for L1 and L2 regularization, which can be
        applied to both weights and biases to prevent overfitting.

        Parameters:
            num_inputs (int): The number of input features (i.e., the number of
                              neurons in the previous layer).
            num_neurons (int): The number of neurons in the current layer.
            weight_regularizer_l1 (float, optional): L1 regularization factor
                                                     for the weights. Default
                                                     is 0.
            weight_regularizer_l2 (float, optional): L2 regularization factor
                                                     for the weights. Default
                                                     is 0.
            bias_regularizer_l1 (float, optional): L1 regularization factor for
                                                   the biases. Default is 0.
            bias_regularizer_l2 (float, optional): L2 regularization factor for
                                                   the biases. Default is 0.

        Returns:
            None
        """
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training) -> None:
        """
        Performs a forward pass through the dense layer.

        This method calculates the output of the dense layer by performing a
        dot product between the input data and the weight matrix, and then
        adding the bias vector. The output is then stored in the `output`
        attribute.

        Parameters:
            inputs (np.ndarray): The input data or the output from the previous
                                 layer. It should have a shape of (n_samples,
                                 num_inputs).
            training (bool, optional): A flag indicating whether the layer is
                                       in training mode. Default is False.

        Returns:
            None: The result of the forward pass is stored in the `output`
                  attribute.
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the dense layer.

        This method calculates the gradients of the loss with respect to the
        layer's weights, biases, and inputs. These gradients are used during
        the optimization step to update the weights and biases. Regularization
        penalties are also applied if L1 or L2 regularization factors were
        specified.

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
    the input units during the forward pass. During inference, the layer passes
    the inputs through unchanged.

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
        input units to drop (deactivate) during the forward pass. The rate is
        stored as the complement (1 - rate) to facilitate the creation of the
        binary mask during training.

        Parameters:
            rate (float): The dropout rate, a float between 0 and 1. This
                          specifies the fraction of input units to drop.

        Returns:
            None
        """
        self.rate = 1 - rate

    def forward(self, inputs, training) -> None:
        """
        Performs a forward pass through the dropout layer.

        During the forward pass, a binary mask is created using a binomial
        distribution based on the dropout rate. The input values are then
        multiplied by this mask to drop (deactivate) a portion of the inputs
        during training. If the model is not in training mode, the inputs are
        passed through unchanged.

        Parameters:
            inputs (np.ndarray): The input values to the layer. The shape
                                 should match the output of the previous layer.
            training (bool): A flag indicating whether the layer is in training
                             mode. Dropout is only applied during training.

        Returns:
            None: The output after applying dropout is stored in the `output`
                  attribute.
        """
        self.inputs = inputs

        if not training:
            self.output - inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the dropout layer.

        During backpropagation, the gradient of the loss with respect to the
        inputs is calculated by multiplying the incoming gradient by the same
        binary mask used during the forward pass. This ensures that the
        gradient is only propagated through the units that were not dropped
        during the forward pass.

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

    The ReLU activation function is a non-linear function commonly used in
    neural networks. It introduces non-linearity into the model, which helps
    it to learn complex patterns. ReLU operates by setting all negative input
    values to zero while retaining positive input values unchanged. This makes
    it computationally efficient and effective in deep learning models.

    Attributes:
        inputs (np.ndarray): The input values to the activation function during
                             the forward pass.
        output (np.ndarray): The output of the ReLU activation function after
                             applying it to the input data.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs of the activation function, used during
                              backpropagation.
    """

    def forward(self, inputs, training) -> None:
        """
        Performs a forward pass through the ReLU activation function.

        This method applies the ReLU activation function element-wise to the
        input data. It sets any negative values in the input to zero while
        leaving positive values unchanged. This function is typically used in
        hidden layers of a neural network.

        Parameters:
            inputs (np.ndarray): The input data to the activation function. It
                                 should have a shape corresponding to the
                                 output of the previous layer (or the input
                                 data if this is the first layer).
            training (bool, optional): A flag indicating whether the layer is
                                       in training mode. Although ReLU behaves
                                       the same during training and inference,
                                       this parameter is included for
                                       consistency with other layers. Default
                                       is False.

        Returns:
            None: The output of the ReLU activation is stored in the `output`
                  attribute.
        """
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the ReLU activation function.

        During backpropagation, this method calculates the gradient of the loss
        with respect to the inputs of the ReLU function. The gradient is only
        passed through the input values that were positive during the forward
        pass, as the gradient of ReLU is 0 for any input less than or equal to
        0. This helps in efficiently updating the model's weights.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the activation function. It has the
                                  same shape as the `output` attribute.

        Returns:
            None: The gradients are stored in the `dinputs` attribute, ready
                  for backpropagation to the previous layer.
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        """
        Returns the predictions based on the ReLU activation function.

        Since ReLU does not alter positive values and sets negative values to
        zero, the `predictions` method simply returns the outputs as they are.
        This is typically used when ReLU is followed by a layer that produces
        the final predictions in a model.

        Parameters:
            outputs (np.ndarray): The output data from the ReLU activation
                                  function, typically passed from the previous
                                  layer.

        Returns:
            np.ndarray: The predictions, which are the same as the outputs.
        """
        return outputs


class Activation_Softmax:
    """
    Implements the Softmax activation function.

    The Softmax activation function is commonly used in the output layer of a
    neural network for multi-class classification tasks. It converts the logits
    (raw outputs of the network) into a probability distribution, where each
    output represents the probability of a particular class, and the sum of all
    probabilities equals 1. This makes it particularly suitable for tasks where
    the goal is to assign a sample to one of several classes.

    Attributes:
        inputs (np.ndarray): The input data to the activation function,
                             typically the logits from the previous layer.
        output (np.ndarray): The output probabilities after applying the
                             Softmax function to the input data.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs of the activation function, used during
                              backpropagation.
    """

    def forward(self, inputs, training) -> None:
        """
        Performs a forward pass through the Softmax activation function.

        The forward method applies the Softmax function to the input data,
        converting the logits into probabilities. The Softmax function operates
        by computing the exponentials of the input values, then normalizing
        these exponentials by dividing by the sum of all exponentials for each
        sample. This ensures that the outputs form a valid probability
        distribution.

        Parameters:
            inputs (np.ndarray): The input data to the activation function,
                                 typically the logits from the last layer of
                                 the network. The shape should be (n_samples,
                                 n_classes), where n_samples is the number of
                                 samples and n_classes is the number of output
                                 classes.
            training (bool, optional): A flag indicating whether the layer is
                                       in training mode. While this does not
                                       affect the Softmax activation itself,
                                       the parameter is included for
                                       consistency with other layers.
                                       Default is False.

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

        During backpropagation, this method calculates the gradient of the loss
        with respect to the inputs of the Softmax function. This is achieved by
        computing the Jacobian matrix of the Softmax function, which describes
        how each output probability depends on each input logit. The method
        then multiplies this Jacobian matrix by the gradient of the loss with
        respect to the output, yielding the gradient with respect to the input.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the activation function. It has the
                                  same shape as the `output` attribute.

        Returns:
            None: The gradients are stored in the `dinputs` attribute, ready
                  for backpropagation to the previous layer.
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(
                zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        """
        Converts Softmax output probabilities into class predictions.

        This method selects the class with the highest probability for each
        sample as the predicted class. Since the output of the Softmax
        function is a probability distribution over several classes, the index
        with the maximum probability represents the model's prediction for
        that sample.

        Parameters:
            outputs (np.ndarray): The output probabilities from the Softmax
                                  function. The shape should be (n_samples,
                                  n_classes).

        Returns:
            np.ndarray: An array of class predictions (indices) for each
                        sample.
        """
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:
    """
    Implements the Sigmoid activation function.

    The Sigmoid activation function maps input values to an output range
    between 0 and 1. It is commonly used in the output layer of binary
    classification models, where the output represents the probability of the
    positive class. The Sigmoid function is particularly useful in scenarios
    where a decision needs to be made between two classes, such as in logistic
    regression or binary classification neural networks.

    Attributes:
        inputs (np.ndarray): The input values to the activation function during
                             the forward pass.
        output (np.ndarray): The output values after applying the Sigmoid
                             activation function, representing probabilities
                             between 0 and 1.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs, used during backpropagation.
    """

    def forward(self, inputs, training) -> None:
        """
        Performs a forward pass through the Sigmoid activation function.

        This method applies the Sigmoid function to each element in the input
        array, transforming input values into probabilities between 0 and 1.
        The Sigmoid function is defined as 1 / (1 + exp(-x)), which ensures
        that the output is always within the range [0, 1].

        Parameters:
            inputs (np.ndarray): The input data to the activation function,
                                 typically the output from the previous layer
                                 of the network.
            training (bool, optional): A flag indicating whether the layer is
                                       in training mode. Although the Sigmoid
                                       function itself behaves the same during
                                       training and inference, this parameter
                                       is included for consistency with other
                                       layers. Default is False.

        Returns:
            None: The output of the Sigmoid function is stored in the `output`
                  attribute, representing the predicted probabilities.
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the Sigmoid activation function.

        During backpropagation, this method calculates the gradient of the loss
        with respect to the inputs by applying the derivative of the Sigmoid
        function. The derivative of the Sigmoid function is given by
        sigmoid(x) * (1 - sigmoid(x)), which is used to compute how the loss
        changes with respect to the inputs. This gradient is then used to
        update the model's weights.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the Sigmoid activation function.

        Returns:
            None: The gradients after applying the Sigmoid derivative are
                  stored in the `dinputs` attribute, ready for backpropagation
                  to the previous layer.
        """
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        """
        Converts the Sigmoid output probabilities into binary predictions.

        This method applies a threshold of 0.5 to the outputs of the Sigmoid
        function to determine the predicted class. If the output is greater
        than 0.5, the prediction is 1 (positive class); otherwise, it is 0
        (negative class). This threshold-based approach is common in binary
        classification tasks where the goal is to decide between two possible
        outcomes.

        Parameters:
            outputs (np.ndarray): The output values from the Sigmoid function,
                                  typically representing probabilities between
                                  0 and 1 for each sample.

        Returns:
            np.ndarray: Binary predictions (0 or 1) based on the Sigmoid
                        outputs, where 1 indicates the positive class and 0
                        indicates the negative class.
        """
        return (outputs > 0.5) * 1


class Activation_Linear:
    """
    Implements a linear activation function.

    The linear activation function, also known as the identity function, simply
    returns the input as output. It is often used in the output layer of
    regression models where the output is a continuous value rather than a
    probability. This function is essential in scenarios where the network's
    output needs to represent a direct mapping of the input, particularly in
    tasks involving prediction of real-valued numbers.

    Attributes:
        inputs (np.ndarray): The input values to the activation function during
                             the forward pass.
        output (np.ndarray): The output values, which in the case of the linear
                             activation function, are identical to the inputs.
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs, used during backpropagation.
    """

    def forward(self, inputs, training) -> None:
        """
        Performs a forward pass through the linear activation function.

        In the linear activation function, the output is directly equal to the
        input. This function is typically used in the output layer of
        regression models, where the model is predicting continuous values.

        Parameters:
            inputs (np.ndarray): The input data to the activation function.
                                 This data typically comes from the previous
                                 layer of the network.
            training (bool, optional): A flag indicating whether the layer is
                                       in training mode. Although the linear
                                       activation function behaves the same
                                       during training and inference, this
                                       parameter is included for consistency
                                       with other layers. Default is False.

        Returns:
            None: The output is stored in the `output` attribute, which is the
                  same as the input.
        """
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues) -> None:
        """
        Performs a backward pass through the linear activation function.

        During backpropagation, the gradient of the loss with respect to the
        inputs is passed through unchanged because the derivative of the linear
        function is 1. This means that the error is propagated backward without
        any modification, which is necessary for updating the model's weights
        correctly.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the linear activation function.
                                  This is typically the gradient coming from
                                  the loss function or the next layer in the
                                  network.

        Returns:
            None: The gradients are stored in the `dinputs` attribute, ready
                  for backpropagation to the previous layer.
        """
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        """
        Returns the predictions made by the model.

        For a linear activation function, the predictions are simply the output
        values themselves. This method is particularly useful in regression
        models, where the network's final layer outputs continuous values that
        represent the model's predictions.

        Parameters:
            outputs (np.ndarray): The output values from the final layer of the
                                  model, typically representing predicted
                                  continuous values.

        Returns:
            np.ndarray: The predictions, which are the same as the outputs in
                        this case.
        """
        return outputs


class Loss:
    """
    Base class for loss functions.

    This class serves as a base class for various loss functions used in
    neural networks. It provides a method to calculate the average loss over a
    batch of samples and also handles regularization loss. The `forward`
    method, which computes the loss for each sample, should be implemented in
    subclasses.

    Attributes:
        trainable_layers (list): A list of layers in the model that have
                                 trainable parameters (weights and biases).
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
        for layer in self.trainable_layers:

            if layer.weight_regularization_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(
                    np.abs(layer.weights))

            if layer.weight_regularization_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(
                    layer.weights * layer.weights)

            if layer.bias_regularization_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(
                    np.abs(layer.bias))

            if layer.bias_regularization_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(
                    layer.bias * layer.bias)

        return regularization_loss

    def calculate(self, output, y, *, include_regularization=False):
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
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def remember_trainable_layers(self, trainable_layers):
        """
        Stores a reference to the trainable layers in the model.

        This method is used to pass a reference of the trainable layers to the
        loss function, so that the loss function can calculate regularization
        loss on those layers.

        Parameters:
            trainable_layers (list): A list of layers that have trainable
                                     parameters (weights and biases).

        Returns:
            None
        """
        self.trainable_layers = trainable_layers


class Loss_CategoricalCrossEntropy(Loss):
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


class LossBinaryCrossentropy(Loss):
    """
    Implements the binary cross-entropy loss function.

    This loss function is commonly used in binary classification tasks. It
    measures the difference between two probability distributions,
    specifically the predicted probabilities and the actual binary class
    labels. The loss function is designed to output a loss value that reflects
    how well the model's predictions align with the true labels.

    Attributes:
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs, used during backpropagation.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the binary cross-entropy loss.

        This method calculates the binary cross-entropy loss by comparing the
        predicted probabilities with the true binary class labels. The method
        ensures numerical stability by clipping the predicted probabilities to
        avoid taking the log of zero.

        Parameters:
            y_pred (np.ndarray): The predicted probabilities from the model.
                                 The shape should be (n_samples, n_outputs).
            y_true (np.ndarray): The true binary class labels. This should
                                 have the same shape as `y_pred`.

        Returns:
            np.ndarray: An array of loss values for each sample in the batch.
        """
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Performs a backward pass through the binary cross-entropy loss.

        This method calculates the gradient of the loss with respect to the
        inputs (the predicted probabilities). This gradient is used during
        backpropagation to update the model's weights.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the previous layer (or activation
                                  function). The shape should be
                                  (n_samples, n_outputs).
            y_true (np.ndarray): The true binary class labels. This should
                                  have the same shape as `dvalues`.

        Returns:
            None: The gradients are stored in the `dinputs` attribute.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


class Loss_MeanSquareError(Loss):
    """
    Implements the Mean Squared Error (MSE) loss function.

    The Mean Squared Error (MSE) loss function is commonly used in regression
    tasks. It measures the average squared difference between the predicted
    values and the true values, providing a measure of how well the model's
    predictions align with the true continuous values.

    Attributes:
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs,used during backpropagation.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the Mean Squared Error (MSE) loss.

        This method calculates the MSE loss by comparing the predicted values
        with the true values. The loss is computed as the average of the
        squared differences between the predicted and true values for each
        sample.

        Parameters:
            y_pred (np.ndarray): The predicted values from the model.
                                 The shape should be (n_samples, n_outputs).
            y_true (np.ndarray): The true continuous values. This should have
                                 the same shape as `y_pred`.

        Returns:
            np.ndarray: An array of loss values for each sample in the batch.
        """
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Performs a backward pass through the Mean Squared Error (MSE) loss.

        This method calculates the gradient of the loss with respect to the
        inputs (the predicted values). This gradient is used during
        backpropagation to update the model's weights.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the previous layer (or activation
                                  function). The shape should be
                                  (n_samples, n_outputs).
            y_true (np.ndarray): The true continuous values. This should have
                                 the same shape as `dvalues`.

        Returns:
            None: The gradients are stored in the `dinputs` attribute.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):
    """
    Implements the Mean Absolute Error (MAE) loss function.

    The Mean Absolute Error (MAE) loss function is commonly used in regression
    tasks. It measures the average absolute difference between the predicted
    values and the true values, providing a measure of how well the model's
    predictions align with the true continuous values.

    Attributes:
        dinputs (np.ndarray): The gradient of the loss with respect to the
                              inputs, used during backpropagation.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the Mean Absolute Error (MAE) loss.

        This method calculates the MAE loss by comparing the predicted values
        with the true values. The loss is computed as the average of the
        absolute differences between the predicted and true values for each
        sample.

        Parameters:
            y_pred (np.ndarray): The predicted values from the model.
                                 The shape should be (n_samples, n_outputs).
            y_true (np.ndarray): The true continuous values. This should have
                                 the same shape as `y_pred`.

        Returns:
            np.ndarray: An array of loss values for each sample in the batch.
        """
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Performs a backward pass through the Mean Absolute Error (MAE) loss.

        This method calculates the gradient of the loss with respect to the
        inputs (the predicted values). This gradient is used during
        backpropagation to update the model's weights.

        Parameters:
            dvalues (np.ndarray): The gradient of the loss with respect to the
                                  output of the previous layer (or activation
                                  function). The shape should be
                                  (n_samples, n_outputs).
            y_true (np.ndarray): The true continuous values. This should have
                                 the same shape as `dvalues`.

        Returns:
            None: The gradients are stored in the `dinputs` attribute.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
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

    def __init__(self, learning_rate=1.0, decay=0., momentum=0.) -> None:
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
    algorithm that combines the advantages of two other popular methods:
    AdaGrad and RMSProp. It computes adaptive learning rates for each parameter
    by keeping track of the first and second moments of the gradients, making
    it suitable for problems with large data and/or many parameters.

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

        This method sets up the optimizer with the specified learning rate,
        decay, epsilon, beta_1, and beta_2 parameters. It also initializes
        the current learning rate and iteration count.

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

        layer.weights += (
            -self.current_learning_rate * weight_momentums_corrected /
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate * bias_momentums_corrected /
            (np.sqrt(bias_cache_corrected) + self.epsilon)
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


class Layer_Input:
    """
    Represents the input layer of a neural network.

    The input layer serves as the entry point for data into the neural network.
    It does not perform any computations but simply passes the input data
    forward to the next layer in the network. This layer is essential for
    defining the input shape for the model, ensuring that subsequent layers
    receive data in the correct format.

    Attributes:
        output (np.ndarray): The data that is passed to the next layer in the
                             network. This is simply a copy of the input data
                             provided to the layer.
    """

    def forward(self, inputs, training):
        """
        Performs a forward pass through the input layer.

        The forward method for the input layer simply sets the input data as
        the output. This output will then be passed to the next layer in the
        model. The method supports both training and inference modes, although
        it behaves identically in both cases since no transformations are
        applied to the input data.

        Parameters:
            inputs (np.ndarray): The input data provided to the neural network.
                                 This data should have a shape consistent with
                                 the expected input shape of the model.
            training (bool, optional): A flag indicating whether the model is
                                       in training mode. This parameter is
                                       included to maintain a consistent
                                       interface with other layers, but it does
                                       not affect the behavior of the input
                                       layer. Default is False.

        Returns:
            None: The input data is stored in the `output` attribute and
                  passed to the next layer in the network.
        """
        self.output = inputs


class Model:
    """
    Represents a neural network model.

    This class encapsulates a neural network model with methods to add layers,
    set loss functions, optimizers, and accuracy metrics, and train the model
    on data. It supports various activation functions and loss functions, and
    can be extended to include custom layers or metrics.

    Attributes:
        layers (list): A list of layers in the neural network.
        softmax_classifier_output (object): An object combining Softmax
                                            activation and categorical cross-
                                            entropy loss for optimization.
    """

    def __init__(self) -> None:
        """
        Initializes the Model instance.

        This constructor initializes the layers list and sets the softmax
        classifier output to None.
        """
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer) -> None:
        """
        Adds a layer to the model.

        This method appends a given layer to the model's list of layers.

        Parameters:
            layer (object): The layer to be added to the model.

        Returns:
            None
        """
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        """
        Sets the loss function, optimizer, and accuracy metric for the model.

        This method configures the model with a specified loss function,
        optimizer, and accuracy metric, which will be used during training.

        Parameters:
            loss (object): The loss function to be used.
            optimizer (object): The optimizer to be used for updating weights.
            accuracy (object): The accuracy metric to evaluate model
                               performance.

        Returns:
            None
        """
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self) -> None:
        """
        Finalizes the model structure before training.

        This method sets up connections between layers and prepares the model
        for training by initializing input layers and handling special cases
        such as softmax classifiers with categorical cross-entropy loss.

        Returns:
            None
        """
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(
                self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = (
                Activation_Softmax_Loss_CategoricalCrossentropy()
            )

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        """
        Trains the model on the given data.

        This method trains the model using the specified number of epochs,
        updating the model's weights using backpropagation and the optimizer.
        It also prints training progress and evaluates the model on validation
        data if provided.

        Parameters:
            X (np.ndarray): The input data for training.
            y (np.ndarray): The true labels corresponding to the input data.
            epochs (int, optional): The number of epochs to train for.
                                    Default is 1.
            print_every (int, optional): Frequency of printing training
                                         progress. Default is 1.
            validation_data (tuple, optional): A tuple (X_val, y_val)
                                               containing validation data and
                                               labels. Default is None.

        Returns:
            None
        """

        self.accuracy.init(y)

        for epoch in range(1, epochs + 1):
            output = self.forward(X, training=True)
            data_loss, regularization_loss = self.loss.calculate(
                output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)
            self.optimizer.pre_update_params()

            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

        if validation_data is not None:
            X_val, y_val = validation_data
            output = self.forward(X_val, training=False)
            loss = self.loss.calculate(output, y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

    def forward(self, X, training) -> np.ndarray:
        """
        Performs a forward pass through the model.

        This method processes the input data through each layer in the model,
        producing the final output.

        Parameters:
            X (np.ndarray): The input data.
            training (bool): A flag indicating whether the model is in training
                             mode.

        Returns:
            np.ndarray: The output of the model after processing through all
                        layers.
        """
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return self.layers.output

    def backward(self, output, y):
        """
        Performs a backward pass through the model.

        This method calculates the gradients for each layer by propagating the
        error backward through the model, updating the model's parameters.

        Parameters:
            output (np.ndarray): The predicted output of the model.
            y (np.ndarray): The true labels corresponding to the input data.

        Returns:
            None
        """
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


class Accuracy:
    """
    Base class for calculating accuracy.

    This class provides a framework for calculating the accuracy of predictions
    made by a model. It serves as a base class for different types of accuracy
    calculations, such as for regression or categorical classification tasks.

    Methods:
        calculate(predictions, y):
            Calculates the accuracy by comparing predictions to the true
            labels.
    """
    def calculate(self, predictions, y):
        """
        Calculates the accuracy of predictions compared to the true labels.

        This method computes the accuracy by comparing the model's predictions
        with the true values and calculating the mean of correct predictions.
        The comparison logic is defined in the `compare` method, which should
        be implemented in subclasses.

        Parameters:
            predictions (np.ndarray): The predicted values from the model.
            y (np.ndarray): The true labels or values.

        Returns:
            float: The accuracy as a proportion of correct predictions.
        """
        comparisons = self.compare(predictions, y)
        accuracy = np.mean*comparisons
        return accuracy


class Accuracy_Regresson(Accuracy):
    """
    Accuracy metric for regression tasks.

    This class extends the `Accuracy` base class to provide an accuracy metric
    for regression tasks. The accuracy is calculated based on the precision,
    which is defined as a fraction of the standard deviation of the true
    values. The precision determines the tolerance for considering a
    prediction to be "accurate."

    Attributes:
        precision (float): The precision threshold used to determine the
                           accuracy of predictions.
    """
    def __init__(self) -> None:
        """
        Initializes the Accuracy_Regression instance.

        This constructor initializes the `precision` attribute to `None`. The
        precision will be set during the first call to `init`, based on the
        standard deviation of the true values.
        """
        self.precision = None

    def init(self, y, reinit=False):
        """
        Initializes or reinitializes the precision threshold.

        This method calculates the precision threshold based on the standard
        deviation of the true values (`y`). The precision is used to determine
        whether a prediction is accurate within a certain range of the true
        value.

        Parameters:
            y (np.ndarray): The true values used to calculate the precision.
            reinit (bool, optional): A flag indicating whether to reinitialize
                                     the precision even if it has already been
                                     set. Default is `False`.

        Returns:
            None
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """
        Compares predictions to true values based on the precision threshold.

        This method checks whether the absolute difference between each
        prediction and the true value is within the defined precision
        threshold. If the difference is less than the precision, the
        prediction is considered accurate.

        Parameters:
            predictions (np.ndarray): The predicted values from the model.
            y (np.ndarray): The true values to compare against.

        Returns:
            np.ndarray: A boolean array indicating whether each prediction is
                        within the precision threshold.
        """
        return np.absolute(predictions - y) < self.precision


class Accuracy_Categorical(Accuracy):
    """
    Accuracy metric for categorical classification tasks.

    This class extends the `Accuracy` base class to provide an accuracy metric
    for categorical classification tasks. It supports both binary and
    multi-class classification. In multi-class classification, predictions are
    compared to the true class labels, with the option to handle one-hot
    encoded labels.

    Attributes:
        binary (bool): Indicates whether the classification task is binary.
    """
    def __init__(self, *, binary=False) -> None:
        """
        Initializes the Accuracy_Categorical instance.

        This constructor initializes the `binary` attribute, which determines
        whether the accuracy calculation should be performed for binary
        classification or multi-class classification.

        Parameters:
            binary (bool, optional): A flag indicating whether the
                                     classification task is binary. Default is
                                     `False`.
        """
        self.binary = binary

    def init(self, y):
        """
        Initializes the accuracy metric for categorical tasks.

        This method is a placeholder for any initialization needed for
        categorical accuracy. In this case, no initialization is required, so
        the method does nothing.

        Parameters:
            y (np.ndarray): The true class labels.

        Returns:
            None
        """
        pass

    def compare(self, predictions, y):
        """
        Compares predictions to true class labels.

        This method compares the predicted class labels to the true class
        labels. For multi-class classification, if the true labels are one-hot
        encoded, they are first converted to class indices before comparison.
        In binary classification, the predictions are directly compared to the
        true labels.

        Parameters:
            predictions (np.ndarray): The predicted class labels from the
                                      model.
            y (np.ndarray): The true class labels.

        Returns:
            np.ndarray: A boolean array indicating whether each prediction
                        matches the true label.
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


# print("Python:", sys.version)
# print("Numpy:", np.__version__)
# print("Matplotlib:", matplotlib.__version__)
