from unittest.mock import patch, MagicMock

import numpy as np

from aidefender.robustness import robustness_accuracy


@patch("art.attacks.Attack.estimator_requirements", return_value=object)
@patch("art.attacks.evasion.FastGradientMethod.generate", side_effect=lambda x: x)
def test_robustness_accuracy_ideal_classifier(mock_generate, mock_estimator_requirements):
    classifier = MagicMock()
    classifier.predict = MagicMock(side_effect=lambda x: np.zeros((len(x), 1)))

    score = robustness_accuracy(classifier, np.zeros((20, 224, 224, 3)), attack_name='fgsm')

    assert isinstance(score, float)
    assert np.isclose(score, 1)


@patch("art.attacks.Attack.estimator_requirements", return_value=object)
@patch("art.attacks.evasion.FastGradientMethod.generate", side_effect=lambda x: x + np.random.random((len(x), 2)))
def test_robustness_accuracy_random_classifier(mock_generate, mock_estimator_requirements):
    classifier = MagicMock()
    classifier.predict = MagicMock(side_effect=lambda x: x)

    score = robustness_accuracy(classifier, np.random.random((200, 2)), attack_name='fgsm')

    assert isinstance(score, float)
    assert score < 1
    assert score >= 0
