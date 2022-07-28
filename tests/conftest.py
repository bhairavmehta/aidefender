import pytest
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication, AuthenticationException

CATS_AND_DOGS_DATA_PATH = "artifacts/data/cats_dogs_small/"
CUSTOMVISION_CATS_AND_DOGS_TF_MODEL_PATH = "artifacts/models/customvision_cats_and_dogs_tf/"
MLFLOW_CATS_AND_DOGS_PYTORCH_MODEL_PATH = "artifacts/models/mlflow_cats_and_dogs_pytorch/"
MLFLOW_CATS_AND_DOGS_TF_MODEL_PATH = "artifacts/models/mlflow_cats_and_dogs_customvision_tf/"

AML_MODEL_NAME = 'customvision_cats_and_dogs_tf'
AML_DATASET_NAME = 'cats_and_dogs_small'


@pytest.fixture(scope="module")
def cats_and_dogs_data_path():
    return CATS_AND_DOGS_DATA_PATH


@pytest.fixture(scope="module")
def customvision_cats_and_dogs_tf_model_path():
    return CUSTOMVISION_CATS_AND_DOGS_TF_MODEL_PATH


@pytest.fixture(scope="module")
def mlflow_cats_and_dogs_pytorch_model_path():
    return MLFLOW_CATS_AND_DOGS_PYTORCH_MODEL_PATH


@pytest.fixture(scope="module")
def mlflow_cats_and_dogs_tf_model_path():
    return MLFLOW_CATS_AND_DOGS_TF_MODEL_PATH


@pytest.fixture(scope="module")
def aml_workspace():
    try:
        ws = Workspace(
            subscription_id="9e7c2d63-bc69-4abc-a8c4-7b90cf90b7de",
            resource_group="rgpmaidapaidefender",
            workspace_name="aidefenderaml",
            auth=AzureCliAuthentication(),
        )
        return ws
    except AuthenticationException:
        return None


@pytest.fixture(scope="module")
def aml_model_name():
    return AML_MODEL_NAME


@pytest.fixture(scope="module")
def aml_dataset_name():
    return AML_DATASET_NAME
