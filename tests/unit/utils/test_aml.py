import os

import pytest

from aidefender.utils.aml import mount_model, mount_dataset


@pytest.mark.parametrize(
    ("workspace", "model_name", ),
    [
        (pytest.lazy_fixture("aml_workspace"),
         pytest.lazy_fixture("aml_model_name")),  # AML case
        (None, pytest.lazy_fixture(
            "customvision_cats_and_dogs_tf_model_path")),  # Local case
    ]
)
def test_mount_model(workspace, model_name):
    # skip if AML is not available
    if workspace is None and not os.path.exists(model_name):
        pytest.skip("AML auth has failed")

    data_path_saved = None
    with mount_model(workspace, model_name) as data_path:
        data_path_saved = data_path

        assert os.path.exists(data_path_saved)

    # check that the path does not exist anymore
    if workspace:
        assert not os.path.exists(data_path_saved)


def test_mount_model_none_ws(aml_model_name):
    with pytest.raises(ValueError):
        with mount_model(None, aml_model_name):
            pass


@pytest.mark.parametrize(
    ("workspace", "dataset_name", ),
    [
        (pytest.lazy_fixture("aml_workspace"),
         pytest.lazy_fixture("aml_dataset_name")),  # AML case
        (None, pytest.lazy_fixture("cats_and_dogs_data_path")),  # Local case
    ]
)
def test_mount_dataset(workspace, dataset_name):
    # skip if AML is not available
    if workspace is None and not os.path.exists(dataset_name):
        pytest.skip("AML auth has failed")

    data_path_saved = None
    with mount_dataset(workspace, dataset_name) as data_path:
        data_path_saved = data_path

        assert os.path.exists(data_path_saved)

    # check that the path does not exist anymore
    if workspace:
        assert not os.path.exists(data_path_saved)


def test_mount_dataset_none_ws(aml_dataset_name):
    with pytest.raises(ValueError):
        with mount_model(None, aml_dataset_name):
            pass
