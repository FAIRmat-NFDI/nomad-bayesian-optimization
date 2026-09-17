"""Unit tests for the conversion of serialized BayBE campaigns into schema dicts.

These tests operate on plain campaign dicts. Only the acquisition function
abbreviations are looked up from BayBE, so those assertions are skipped when BayBE
is not installed.
"""

import importlib.util

import pytest

from nomad_bayesian_optimization.campaign_converter import (
    campaign_dict_to_schema_dict,
    discrete_candidate_count,
)

HAS_BAYBE = importlib.util.find_spec('baybe') is not None

GP_SURROGATE = {
    'type': 'CompositeSurrogate',
    'surrogates': {
        'type': '_ReplicationMapping',
        'template': {
            'type': 'GaussianProcessSurrogate',
            'kernel_or_factory': {'type': 'BayBEKernelFactory'},
        },
    },
}


def _campaign(recommender, objective_type='SingleTargetObjective'):
    return {
        'searchspace': {
            'discrete': {'parameters': []},
            'continuous': {'parameters': [{'name': 't'}]},
        },
        'objective': {'type': objective_type, 'target': {'name': 'y'}},
        'recommender': recommender,
        'n_batches_done': 2,
    }


def test_meta_recommender():
    """Nested recommenders, surrogate and default acquisition function are set."""
    recommender = {
        'type': 'TwoPhaseMetaRecommender',
        'initial_recommender': {'type': 'RandomRecommender'},
        'recommender': {
            'type': 'BotorchRecommender',
            'surrogate_model': GP_SURROGATE,
            'acquisition_function': None,
            'hybrid_sampler': None,
            'sampling_percentage': 1.0,
        },
        'switch_after': 1,
    }
    result = campaign_dict_to_schema_dict(_campaign(recommender))

    assert result['search_space_type'] == 'Continuous'
    assert result['n_batches_done'] == 2

    converted = result['recommender']
    assert converted['type'] == 'TwoPhaseMetaRecommender'
    assert converted['config'] == recommender
    assert converted['switch_after'] == 1
    assert converted['initial_recommender']['type'] == 'RandomRecommender'
    assert 'acquisition_function' not in converted['initial_recommender']

    bayesian = converted['recommender']
    assert 'config' not in bayesian
    assert 'hybrid_sampler' not in bayesian
    assert bayesian['sampling_percentage'] == 1.0
    assert bayesian['surrogate_model']['type'] == 'GaussianProcessSurrogate'
    assert bayesian['surrogate_model']['kernel_factory']['type'] == (
        'BayBEKernelFactory'
    )
    assert bayesian['acquisition_function']['type'] == 'qLogExpectedImprovement'
    if HAS_BAYBE:
        assert bayesian['acquisition_function']['abbreviation'] == 'qLogEI'


def test_plain_bayesian_recommender():
    """A non-meta recommender stores its parts on the top-level recommender."""
    recommender = {
        'type': 'BotorchRecommender',
        'surrogate_model': {'type': 'RandomForestSurrogate', 'model_params': {}},
        'acquisition_function': {'type': 'qUpperConfidenceBound', 'beta': 0.2},
    }
    converted = campaign_dict_to_schema_dict(_campaign(recommender))['recommender']

    assert converted['surrogate_model']['type'] == 'RandomForestSurrogate'
    assert 'kernel_factory' not in converted['surrogate_model']
    assert converted['acquisition_function']['type'] == 'qUpperConfidenceBound'
    if HAS_BAYBE:
        assert converted['acquisition_function']['abbreviation'] == 'qUCB'


def test_pareto_default_acquisition_function():
    """Pareto objectives default to a hypervolume-based acquisition function."""
    recommender = {'type': 'BotorchRecommender', 'acquisition_function': None}
    converted = campaign_dict_to_schema_dict(
        _campaign(recommender, objective_type='ParetoObjective')
    )['recommender']

    acquisition_function = converted['acquisition_function']
    assert acquisition_function['type'] == 'qLogNoisyExpectedHypervolumeImprovement'
    if HAS_BAYBE:
        assert acquisition_function['abbreviation'] == 'qLogNEHVI'


def test_random_recommender():
    """Non-Bayesian recommenders have no surrogate or acquisition function."""
    campaign = _campaign({'type': 'RandomRecommender'})
    converted = campaign_dict_to_schema_dict(campaign)['recommender']

    assert converted['type'] == 'RandomRecommender'
    assert 'surrogate_model' not in converted
    assert 'acquisition_function' not in converted


@pytest.mark.parametrize(
    'discrete, continuous, expected',
    [
        pytest.param([{'name': 'p'}], [], 'Discrete', id='discrete'),
        pytest.param([], [{'name': 't'}], 'Continuous', id='continuous'),
        pytest.param([{'name': 'p'}], [{'name': 't'}], 'Hybrid', id='hybrid'),
    ],
)
def test_search_space_type(discrete, continuous, expected):
    campaign = {
        'searchspace': {
            'discrete': {'parameters': discrete},
            'continuous': {'parameters': continuous},
        }
    }
    assert campaign_dict_to_schema_dict(campaign)['search_space_type'] == expected


def test_no_discrete_candidates_for_continuous_search_space():
    assert discrete_candidate_count(_campaign({'type': 'RandomRecommender'})) is None


def _legacy_match_target(T):
    with pytest.warns(DeprecationWarning):
        return T('y', 'MATCH', bounds=(1.0, 3.0))


@pytest.mark.skipif(not HAS_BAYBE, reason='BayBE is required to create targets.')
@pytest.mark.parametrize(
    'make_target, expected',
    [
        pytest.param(lambda T: T('y'), {'mode': 'MAX'}, id='max'),
        pytest.param(lambda T: T('y', minimize=True), {'mode': 'MIN'}, id='min'),
        pytest.param(
            lambda T: T.match_bell('y', match_value=2.0, sigma=0.2),
            {'mode': 'MATCH', 'match_value': 2.0, 'match_mode': '='},
            id='bell',
        ),
        pytest.param(
            lambda T: T.match_triangular('y', cutoffs=(1.0, 3.0)),
            {'mode': 'MATCH', 'match_value': 2.0, 'match_mode': '='},
            id='triangular-cutoffs',
        ),
        pytest.param(
            lambda T: T.match_absolute('y', match_value=2.0),
            {'mode': 'MATCH', 'match_value': 2.0, 'match_mode': '='},
            id='absolute',
        ),
        pytest.param(
            lambda T: T.match_bell('y', match_value=2.0, sigma=0.2, match_mode='>='),
            {'mode': 'MATCH', 'match_value': 2.0, 'match_mode': '>='},
            id='bell-geq',
        ),
        pytest.param(
            lambda T: T.match_bell(
                'y', match_value=2.0, sigma=0.2, mismatch_instead=True
            ),
            {'mode': 'MISMATCH', 'match_value': 2.0, 'match_mode': '='},
            id='bell-mismatch',
        ),
        pytest.param(
            _legacy_match_target,
            {'mode': 'MATCH', 'match_value': 2.0, 'match_mode': '='},
            id='legacy-match',
        ),
        pytest.param(
            lambda T: T.normalized_sigmoid('y', anchors=((0, 0.1), (10, 0.9))),
            {},
            id='sigmoid',
        ),
    ],
)
def test_target_goal(make_target, expected):
    """The goal of a target is recorded regardless of how BayBE transforms it."""
    import json

    from baybe.objectives import SingleTargetObjective
    from baybe.targets import NumericalTarget

    campaign = _campaign({'type': 'RandomRecommender'})
    target = make_target(NumericalTarget)
    campaign['objective'] = json.loads(SingleTargetObjective(target).to_json())
    converted = campaign_dict_to_schema_dict(campaign)['objective']['targets'][0]

    for key in ('mode', 'match_value', 'match_mode'):
        assert converted.get(key) == expected.get(key)
