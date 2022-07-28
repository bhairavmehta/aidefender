import numpy as np
import pytest


from aidefender.utils.data import load_image_dataset


def test_load_image_dataset(cats_and_dogs_data_path):
    nb_samples = 20
    target_size = 107
    labels = ['cat', 'dog']
    file_format = 'jpg'

    images, labels = load_image_dataset(
        cats_and_dogs_data_path, labels, target_size, file_format)

    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert images.dtype == np.uint8
    assert labels.dtype == np.long
    assert images.shape == (nb_samples, target_size, target_size, 3)
    assert labels.shape == (nb_samples, )


def test_load_cats_and_dogs_wrong_dir(tmp_path):
    target_size = 107
    labels = ['cat', 'dog']
    file_format = 'jpg'

    with pytest.raises(ValueError):
        _, _ = load_image_dataset(tmp_path, labels, target_size, file_format)


def test_load_cats_and_dogs_wrong_format(cats_and_dogs_data_path):
    target_size = 107
    labels = ['cat', 'dog']
    file_format = 'png'

    with pytest.raises(ValueError):
        _, _ = load_image_dataset(
            cats_and_dogs_data_path, labels, target_size, file_format)
