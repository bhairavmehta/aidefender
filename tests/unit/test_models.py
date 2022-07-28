import pytest
import numpy as np
from art.classifiers import TensorFlowClassifier

from aidefender.models import CustomVisionTensorFlowModel


@pytest.mark.parametrize(
    ("model_class", "model_path", "target_classifier_class"),
    [
        (
            CustomVisionTensorFlowModel, pytest.lazy_fixture(
                "customvision_cats_and_dogs_tf_model_path"),
            TensorFlowClassifier
        ),
    ]
)
def test_init(model_class, model_path, target_classifier_class):
    model = model_class(model_path)

    assert isinstance(model.classifier, target_classifier_class)


@pytest.mark.parametrize(
    ("model_class", "model_path"),
    [
        (CustomVisionTensorFlowModel, pytest.lazy_fixture(
            "customvision_cats_and_dogs_tf_model_path")),
    ]
)
def test_predict(model_class, model_path):
    model = model_class(model_path)

    nb_samples = 3
    target_size = 224
    nb_channels = 3
    nb_classes = 2
    images = np.random.randint(
        1,
        255,
        size=(
            nb_samples,
            target_size,
            target_size,
            nb_channels),
        dtype=np.uint8)

    preds = model.predict(images)

    assert isinstance(preds, np.ndarray)
    assert preds.dtype == np.float32
    assert preds.shape == (nb_samples, nb_classes)
