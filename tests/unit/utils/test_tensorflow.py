import os

import tensorflow as tf
from art.estimators.classification import TensorFlowClassifier

from aidefender.utils.tensorflow import load_graph, insert_labels_and_loss, create_art_classifier
from aidefender.utils.tensorflow import get_graph_inputs_names, get_graph_outputs_names
from aidefender.models import CustomVisionTensorFlowModel


def test_load_graph(customvision_cats_and_dogs_tf_model_path):
    model_file = os.path.join(
        customvision_cats_and_dogs_tf_model_path,
        'model.pb')
    graph = load_graph(model_file)

    assert isinstance(graph, tf.Graph)


def test_insert_labels_and_loss(customvision_cats_and_dogs_tf_model_path):
    model_file = os.path.join(
        customvision_cats_and_dogs_tf_model_path,
        'model.pb')
    graph = load_graph(model_file)

    labels, loss = insert_labels_and_loss(
        graph, CustomVisionTensorFlowModel._LOGITS_NODE_NAME,
        CustomVisionTensorFlowModel._LABELS_OP_NAME, CustomVisionTensorFlowModel._LOSS_OP_NAME
    )

    # check that the tensors are in the graph
    labels_tensor = graph.get_tensor_by_name(
        CustomVisionTensorFlowModel._LABELS_NODE_NAME)
    loss_tensor = graph.get_tensor_by_name(
        CustomVisionTensorFlowModel._LOSS_NODE_NAME)

    # check the names
    assert labels.name == CustomVisionTensorFlowModel._LABELS_NODE_NAME
    assert loss.name == CustomVisionTensorFlowModel._LOSS_NODE_NAME

    # check the shape
    assert labels_tensor.get_shape().as_list() == [None, ]
    assert loss_tensor.get_shape().as_list() == []


def test_get_graph_outputs_names(customvision_cats_and_dogs_tf_model_path):
    model_file = os.path.join(
        customvision_cats_and_dogs_tf_model_path,
        'model.pb')
    true_outputs = ['model_outputs']

    graph = load_graph(model_file)
    possible_outputs = get_graph_outputs_names(graph)

    assert isinstance(possible_outputs, list)
    assert possible_outputs == true_outputs


def test_get_graph_inputs_names(customvision_cats_and_dogs_tf_model_path):
    model_file = os.path.join(
        customvision_cats_and_dogs_tf_model_path,
        'model.pb')
    true_inputs = ['Placeholder']

    graph = load_graph(model_file)
    possible_inputs = get_graph_inputs_names(graph)

    assert isinstance(possible_inputs, list)
    assert possible_inputs == true_inputs


def test_create_art_classifier(customvision_cats_and_dogs_tf_model_path):
    model_file = os.path.join(
        customvision_cats_and_dogs_tf_model_path,
        'model.pb')

    graph = load_graph(model_file)
    insert_labels_and_loss(
        graph, CustomVisionTensorFlowModel._LOGITS_NODE_NAME,
        CustomVisionTensorFlowModel._LABELS_OP_NAME, CustomVisionTensorFlowModel._LOSS_OP_NAME
    )

    with tf.compat.v1.Session(graph=graph) as sess:
        classifier = create_art_classifier(
            sess, CustomVisionTensorFlowModel._INPUT_NODE_NAME, CustomVisionTensorFlowModel._LOGITS_NODE_NAME,
            CustomVisionTensorFlowModel._LABELS_NODE_NAME, CustomVisionTensorFlowModel._LOSS_NODE_NAME,
        )

        assert isinstance(classifier, TensorFlowClassifier)
