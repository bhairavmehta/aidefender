import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
import mlflow
from mlflow.pyfunc import PyFuncModel
from art.estimators.classification import PyTorchClassifier, TensorFlowClassifier


from aidefender.utils.mlflow import _get_converter, create_art_model
from aidefender.utils.mlflow import BaseMLFlowConverter, PyTorchMLFlowConverter, TensorFlowMLFlowConverter


class MockIntConverter(BaseMLFlowConverter):
    @classmethod
    def _get_supported_types(cls):
        return (int, )

    def _get_input_output_shape(self, art_model):
        return None, None

    def convert(self, mlflow_model):
        return None

    def save(self, art_model, path):
        return None


def test_BaseMLFlowConverter_can_convert():
    int_converter = MockIntConverter()

    int_model = MagicMock(spec=PyFuncModel)
    int_model._model_impl = 1

    str_model = MagicMock(spec=PyFuncModel)
    str_model._model_impl = 'zzz'

    another_model = MagicMock(spec=PyTorchClassifier)

    assert int_converter.can_convert(int_model)
    assert not int_converter.can_convert(str_model)
    assert not int_converter.can_convert(another_model)


@pytest.mark.parametrize(
    ('model_path', 'converter_class'),
    [
        (pytest.lazy_fixture('mlflow_cats_and_dogs_pytorch_model_path'), PyTorchMLFlowConverter),
        (pytest.lazy_fixture('mlflow_cats_and_dogs_tf_model_path'), TensorFlowMLFlowConverter),
    ]
)
def test__get_converter_mlflow(model_path, converter_class):
    mlflow_model = mlflow.pyfunc.load_model(model_path)
    converter = _get_converter(mlflow_model)

    assert isinstance(converter, converter_class)


@pytest.mark.parametrize(
    ('model', 'converter_class'),
    [
        (MagicMock(spec=PyTorchClassifier), PyTorchMLFlowConverter),
        (MagicMock(spec=TensorFlowClassifier), TensorFlowMLFlowConverter),
    ]
)
def test__get_converter_art(model, converter_class):
    converter = _get_converter(model)

    assert isinstance(converter, converter_class)


def test__get_converter_not_supported():
    mlflow_model = MagicMock()
    with pytest.raises(ValueError):
        _get_converter(mlflow_model)


@pytest.mark.parametrize(
    ('model_path', 'converter_class', 'art_model_class'),
    [
        (pytest.lazy_fixture('mlflow_cats_and_dogs_pytorch_model_path'), PyTorchMLFlowConverter, PyTorchClassifier),
        (pytest.lazy_fixture('mlflow_cats_and_dogs_tf_model_path'), TensorFlowMLFlowConverter, TensorFlowClassifier),
    ]
)
def test_converter_convert(model_path, converter_class, art_model_class):
    converter = converter_class()

    mlflow_model = mlflow.pyfunc.load_model(model_path)
    art_model = converter.convert(mlflow_model)

    assert isinstance(art_model, art_model_class)


@pytest.mark.parametrize(
    ('model_path', 'art_model_class'),
    [
        (pytest.lazy_fixture('mlflow_cats_and_dogs_pytorch_model_path'), PyTorchClassifier),
        (pytest.lazy_fixture('mlflow_cats_and_dogs_tf_model_path'), TensorFlowClassifier),
    ]
)
def test_create_art_model(model_path, art_model_class):
    mlflow_model = mlflow.pyfunc.load_model(model_path)
    art_model = create_art_model(mlflow_model)

    assert isinstance(art_model, art_model_class)


@pytest.mark.parametrize(
    ('model_path', 'converter_class',),
    [
        (pytest.lazy_fixture('mlflow_cats_and_dogs_pytorch_model_path'), PyTorchMLFlowConverter,),
    ]
)
def test_BaseMLFlowConverter_create_mlflow_model_signature(model_path, converter_class):
    converter = converter_class()

    mlflow_model = mlflow.pyfunc.load_model(model_path)
    art_model = converter.convert(mlflow_model)

    signature = converter._create_mlflow_model_signature(art_model)
    assert signature == mlflow_model.metadata.signature


@pytest.mark.parametrize(
    ('model_path', 'converter_class', 'preprocessing_defences_path'),
    [
        (
            pytest.lazy_fixture('mlflow_cats_and_dogs_pytorch_model_path'),
            PyTorchMLFlowConverter,
            os.path.join('data', BaseMLFlowConverter.PREPROCESSING_defenceS_PATH),
        ),
    ]
)
def test_converter_save(model_path, converter_class, preprocessing_defences_path, tmp_path):
    converter = converter_class()

    mlflow_model = mlflow.pyfunc.load_model(model_path)
    art_model = converter.convert(mlflow_model)

    converted_model_path = os.path.join(tmp_path, 'model')
    with patch('builtins.open', mock_open()) as mock:
        converter.save(art_model, converted_model_path)

        mock.assert_any_call(os.path.join(converted_model_path, 'MLmodel'), 'w')
        mock.assert_any_call(os.path.join(converted_model_path, preprocessing_defences_path), 'wb')
