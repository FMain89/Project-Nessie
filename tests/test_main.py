import contextlib
import io
import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import (
    Accuracy_Categorical,
    Accuracy_Regresson,
    Activation_Softmax,
    Layer_Dense,
    Loss_CategoricalCrossEntropy,
    Loss_MeanAbsoluteError,
    Loss_MeanSquareError,
    Model,
    Optimizer_Adagrad,
    Optimizer_Adam,
    Optimizer_RMSprop,
    Optimizer_SGD,
)


class LossTests(unittest.TestCase):
    def test_bias_regularization_uses_dense_biases(self):
        layer = Layer_Dense(
            1,
            2,
            bias_regularizer_l1=0.5,
            bias_regularizer_l2=0.25,
        )
        layer.biases = np.array([[2.0, -3.0]])

        loss = Loss_MeanSquareError()
        loss.remember_trainable_layers([layer])

        expected = 0.5 * 5.0 + 0.25 * 13.0
        self.assertAlmostEqual(loss.regularization_loss(), expected)

    def test_mae_backward_matches_the_loss_derivative(self):
        predictions = np.array([[2.0, -1.0]])
        targets = np.array([[1.0, 1.0]])

        loss = Loss_MeanAbsoluteError()
        loss.backward(predictions, targets)

        np.testing.assert_allclose(loss.dinputs, [[0.5, -0.5]])


class AccuracyTests(unittest.TestCase):
    def test_multioutput_accumulated_accuracy_stays_bounded(self):
        accuracy = Accuracy_Regresson()
        accuracy.precision = 1.0
        accuracy.new_pass()
        predictions = np.array([[0.0, 0.0], [1.0, 1.0]])

        batch_accuracy = accuracy.calculate(predictions, predictions.copy())

        self.assertEqual(batch_accuracy, 1.0)
        self.assertEqual(accuracy.calculate_accumulated(), 1.0)


class OptimizerTests(unittest.TestCase):
    def test_switching_optimizers_keeps_state_independent(self):
        layer = Layer_Dense(2, 2)
        layer.dweights = np.ones_like(layer.weights)
        layer.dbiases = np.ones_like(layer.biases)

        optimizers = (
            Optimizer_SGD(momentum=0.9),
            Optimizer_Adagrad(),
            Optimizer_RMSprop(),
            Optimizer_Adam(),
        )

        for optimizer in optimizers:
            optimizer.pre_update_params()
            optimizer.update_params(layer)
            optimizer.post_update_params()

        self.assertTrue(np.isfinite(layer.weights).all())
        self.assertTrue(np.isfinite(layer.biases).all())


class TrainingTests(unittest.TestCase):
    def test_each_epoch_prints_a_summary_and_runs_validation(self):
        np.random.seed(0)
        model = Model()
        model.add(Layer_Dense(2, 3))
        model.add(Activation_Softmax())
        model.set(
            loss=Loss_CategoricalCrossEntropy(),
            optimizer=Optimizer_SGD(learning_rate=0.1),
            accuracy=Accuracy_Categorical(),
        )
        model.finalize()

        inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
        targets = np.array([0, 1])
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            model.train(
                inputs,
                targets,
                epochs=3,
                validation_data=(inputs, targets),
            )

        self.assertEqual(output.getvalue().count('training,'), 3)
        self.assertEqual(output.getvalue().count('validation,'), 3)


if __name__ == '__main__':
    unittest.main()
