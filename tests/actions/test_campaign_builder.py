"""Tests for the generic spec -> BayBE campaign builder.

``build_campaign`` is the inverse of the parser's
``campaign_dict_to_schema_dict``: it turns a ``BayesianOptimizationInput`` (the
action input describing variables and targets) into a fresh BayBE campaign. The
tests build campaigns of each supported kind, assert the resulting BayBE objects,
and round-trip through the converter so the mapping stays consistent with the
schema the parser produces.
"""

import json

import pytest

pytest.importorskip('baybe')

from nomad_bayesian_optimization.actions.campaign.models import (  # noqa: E402
    BayesianOptimizationInput,
    TargetSpec,
    VariableSpec,
)
from nomad_bayesian_optimization.campaign_builder import build_campaign  # noqa: E402
from nomad_bayesian_optimization.campaign_converter import (  # noqa: E402
    campaign_dict_to_schema_dict,
)


def _input(variables, targets, **kwargs):
    return BayesianOptimizationInput(
        upload_id='u',
        user_id='x',
        schema_name='MySample',
        variables=variables,
        targets=targets,
        **kwargs,
    )


def test_build_mixed_search_space():
    """Continuous, numerical-discrete and categorical variables map correctly."""
    spec = _input(
        variables=[
            VariableSpec(
                name='temperature',
                kind='continuous',
                lower_bound=300.0,
                upper_bound=600.0,
            ),
            VariableSpec(
                name='pressure',
                kind='numerical_discrete',
                values=['1', '2', '3'],
                tolerance=0.0,
            ),
            VariableSpec(
                name='substrate',
                kind='categorical',
                values=['Si', 'SiC', 'GaN'],
                encoding='OHE',
            ),
        ],
        targets=[
            TargetSpec(
                name='refractive_index', mode='MATCH', match_value=2.0, sigma=0.2
            )
        ],
    )
    campaign = build_campaign(spec)

    assert {p.name for p in campaign.parameters} == {
        'temperature',
        'pressure',
        'substrate',
    }
    types = {type(p).__name__ for p in campaign.parameters}
    assert types == {
        'NumericalContinuousParameter',
        'NumericalDiscreteParameter',
        'CategoricalParameter',
    }
    assert type(campaign.objective).__name__ == 'SingleTargetObjective'

    # The campaign can recommend and the converter maps it back onto the schema.
    recommendation = campaign.recommend(batch_size=1)
    assert set(recommendation.columns) == {'temperature', 'pressure', 'substrate'}
    schema_dict = campaign_dict_to_schema_dict(json.loads(campaign.to_json()))
    assert {p['name'] for p in schema_dict['parameters']} == {
        'temperature',
        'pressure',
        'substrate',
    }


@pytest.mark.parametrize(
    'mode,minimize',
    [('MAX', False), ('MIN', True)],
)
def test_single_target_modes(mode, minimize):
    spec = _input(
        variables=[
            VariableSpec(name='t', kind='continuous', lower_bound=0.0, upper_bound=1.0)
        ],
        targets=[TargetSpec(name='y', mode=mode)],
    )
    campaign = build_campaign(spec)
    (target,) = campaign.targets
    assert bool(target.minimize) is minimize


def test_multiple_plain_targets_use_pareto():
    spec = _input(
        variables=[
            VariableSpec(name='t', kind='continuous', lower_bound=0.0, upper_bound=1.0)
        ],
        targets=[
            TargetSpec(name='a', mode='MAX'),
            TargetSpec(name='b', mode='MIN'),
        ],
    )
    campaign = build_campaign(spec)
    assert type(campaign.objective).__name__ == 'ParetoObjective'
    assert {t.name for t in campaign.targets} == {'a', 'b'}


def test_weighted_match_targets_use_desirability():
    spec = _input(
        variables=[
            VariableSpec(name='t', kind='continuous', lower_bound=0.0, upper_bound=1.0)
        ],
        targets=[
            TargetSpec(name='a', mode='MATCH', match_value=0.5, sigma=0.2, weight=1.0),
            TargetSpec(name='b', mode='MATCH', match_value=0.8, sigma=0.2, weight=2.0),
        ],
    )
    campaign = build_campaign(spec)
    assert type(campaign.objective).__name__ == 'DesirabilityObjective'


def test_substance_variable():
    # SubstanceParameter validation requires the optional baybe[chem] stack.
    pytest.importorskip('skfp')
    spec = _input(
        variables=[
            VariableSpec(
                name='solvent',
                kind='substance',
                values=['water=O', 'ethanol=CCO'],
                encoding='RDKIT2DDESCRIPTORS',
            )
        ],
        targets=[TargetSpec(name='solubility', mode='MAX')],
    )
    campaign = build_campaign(spec)
    (parameter,) = campaign.parameters
    assert type(parameter).__name__ == 'SubstanceParameter'
    assert dict(parameter.data) == {'water': 'O', 'ethanol': 'CCO'}


@pytest.mark.parametrize(
    'variables,targets,message',
    [
        (
            [VariableSpec(name='t', kind='continuous')],
            [TargetSpec(name='y', mode='MAX')],
            'lower_bound',
        ),
        (
            [VariableSpec(name='t', kind='numerical_discrete')],
            [TargetSpec(name='y', mode='MAX')],
            'values',
        ),
        (
            [
                VariableSpec(
                    name='t', kind='continuous', lower_bound=0.0, upper_bound=1.0
                )
            ],
            [TargetSpec(name='y', mode='MATCH')],
            'match_value',
        ),
    ],
)
def test_validation_errors(variables, targets, message):
    with pytest.raises(ValueError, match=message):
        build_campaign(_input(variables=variables, targets=targets))
