from unittest.mock import patch, MagicMock

import pytest

from art.attacks.evasion import ProjectedGradientDescent

from aidefender.attacks import ProgressiveAttack


@patch("art.attacks.Attack.estimator_requirements", return_value=object)
def test_progressive_attack_valid_parameters(mock_estimator_requirements):
    attack = ProgressiveAttack(MagicMock())

    assert isinstance(attack.attack, ProjectedGradientDescent)


@pytest.mark.parametrize(
    ("param_name", "param_value", ),
    [
        ("norm", "zzz",),
        ("eps_initial", "inf",),
        ("eps_initial", -1,),
        ("eps_increase", "inf",),
        ("eps_increase", -1,),
        ("batch_size", "inf",),
        ("batch_size", -1,),
    ]
)
@patch("art.attacks.Attack.estimator_requirements", return_value=object)
def test_progressive_attack_invalid_parameters(mock_estimator_requirements, param_name, param_value):
    with pytest.raises(ValueError):
        ProgressiveAttack(MagicMock(), **{param_name: param_value})
