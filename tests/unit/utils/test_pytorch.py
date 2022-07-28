from unittest.mock import MagicMock

import pytest
import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier

from aidefender.utils.pytorch import predict, create_art_classifier


@pytest.mark.skip()
def test_predict():
    data = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.long)
    model = MagicMock(side_effect=lambda x: x)

    predictions = predict(model, data, batch_size=10, normalize=False)

    assert predictions.shape == (2, 3)
    assert predictions.dtype == np.long


@pytest.mark.skip()
def test_predict_batch():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3],
                     [4, 4, 4]], dtype=np.long)
    model = MagicMock(side_effect=lambda x: x)

    predictions = predict(model, data, batch_size=3, normalize=False)

    assert predictions.shape == (4, 3)
    assert predictions.dtype == np.long


@pytest.mark.skip()
def test_predict_normalizer():
    data = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    model = MagicMock(side_effect=lambda x: x)

    predictions = predict(model, data, batch_size=10, normalize=True)

    assert predictions.shape == (2, 3)
    assert predictions.dtype == np.float32
    assert np.all(np.isclose(predictions.sum(axis=1), np.ones((2, ))))


@pytest.mark.skip()
def test_create_art_classifier():
    model = torch.nn.Module()
    model.parameter = torch.nn.Parameter(torch.zeros(10))

    classifier = create_art_classifier(model)

    assert isinstance(classifier, PyTorchClassifier)
