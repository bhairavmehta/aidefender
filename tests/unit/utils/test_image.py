import os

import pytest
from PIL import Image

from aidefender.utils.image import resize_and_crop_image


@pytest.mark.parametrize(
    ("image_file", "target_size", ),
    [
        ('cat/cat.2150.jpg', 224),  # a wide image
        ('cat/cat.11826.jpg', 225),  # a tall image
    ]
)
def test_resize_and_crop_image(
        cats_and_dogs_data_path, image_file, target_size):
    image_path = os.path.join(cats_and_dogs_data_path, image_file)

    image = Image.open(image_path)
    image = resize_and_crop_image(image, target_size)

    assert image.size == (target_size, target_size)
