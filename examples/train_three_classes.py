"""Train Project Nessie on a small, deterministic three-class dataset."""

import contextlib
import io
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (  # noqa: E402
    Accuracy_Categorical,
    Activation_ReLU,
    Activation_Softmax,
    Layer_Dense,
    Loss_CategoricalCrossEntropy,
    Model,
    Optimizer_Adam,
)


def create_dataset(seed=42):
    """Create three clusters and split them into training and test sets."""
    generator = np.random.default_rng(seed)
    centers = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [0.0, 1.0],
    ])

    features = np.vstack([
        generator.normal(center, 0.28, size=(100, 2))
        for center in centers
    ])
    labels = np.repeat(np.arange(len(centers)), 100)
    order = generator.permutation(len(features))
    features = features[order]
    labels = labels[order]

    return features[:240], labels[:240], features[240:], labels[240:]


def create_model():
    """Build a small network with one hidden layer."""
    model = Model()
    model.add(Layer_Dense(2, 16))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(16, 3))
    model.add(Activation_Softmax())
    model.set(
        loss=Loss_CategoricalCrossEntropy(),
        optimizer=Optimizer_Adam(learning_rate=0.02, decay=1e-3),
        accuracy=Accuracy_Categorical(),
    )
    model.finalize()
    return model


def score(model, features, labels):
    """Return categorical cross-entropy loss and classification accuracy."""
    probabilities = model.predict(features)
    losses = model.loss.forward(probabilities, labels)
    predictions = model.output_layer_activation.predictions(probabilities)
    return np.mean(losses), np.mean(predictions == labels), predictions


def main():
    np.random.seed(42)
    train_x, train_y, test_x, test_y = create_dataset()
    model = create_model()

    initial_train_loss, initial_train_accuracy, _ = score(
        model, train_x, train_y)
    initial_test_loss, initial_test_accuracy, _ = score(
        model, test_x, test_y)

    print('Before training')
    print(
        f'  train loss: {initial_train_loss:.4f}, '
        f'accuracy: {initial_train_accuracy:.1%}')
    print(
        f'  test  loss: {initial_test_loss:.4f}, '
        f'accuracy: {initial_test_accuracy:.1%}')

    hidden_training_log = io.StringIO()
    with contextlib.redirect_stdout(hidden_training_log):
        model.train(
            train_x,
            train_y,
            epochs=200,
            batch_size=32,
            print_every=100,
        )

    final_train_loss, final_train_accuracy, _ = score(
        model, train_x, train_y)
    final_test_loss, final_test_accuracy, test_predictions = score(
        model, test_x, test_y)

    print('\nAfter training')
    print(
        f'  train loss: {final_train_loss:.4f}, '
        f'accuracy: {final_train_accuracy:.1%}')
    print(
        f'  test  loss: {final_test_loss:.4f}, '
        f'accuracy: {final_test_accuracy:.1%}')

    print('\nFirst 10 unseen test predictions')
    for expected, predicted in zip(test_y[:10], test_predictions[:10]):
        print(f'  expected {expected}, predicted {predicted}')

    if final_test_accuracy < 0.95:
        raise RuntimeError('The network did not reach the expected accuracy.')


if __name__ == '__main__':
    main()
