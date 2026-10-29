"""Tests for the spin coating specialization of the Bayesian optimization action."""

import pytest
from pydantic import ValidationError

from nomad_bayesian_optimization.actions.spin_coating.models import (
    CONCENTRATION_BOUNDS,
    CONCENTRATION_DECIMALS,
    SCHEMA_NAME,
    SPEED_BOUNDS,
    SPEED_DECIMALS,
    TIME_BOUNDS,
    TIME_DECIMALS,
    SpinCoatingOptimizationInput,
)

BOUNDS = {
    'precursor_concentration': CONCENTRATION_BOUNDS,
    'spin_coating_speed': SPEED_BOUNDS,
    'spinning_time': TIME_BOUNDS,
}
DECIMALS = {
    'precursor_concentration': CONCENTRATION_DECIMALS,
    'spin_coating_speed': SPEED_DECIMALS,
    'spinning_time': TIME_DECIMALS,
}


def _input(**kwargs):
    return SpinCoatingOptimizationInput(
        upload_id='u', user_id='x', target_film_thickness=300.0, **kwargs
    )


def test_to_bayesian_optimization_input():
    """The fixed search space and target are filled in around the user input."""
    spec = _input(batch_size=3, campaign_name='my_campaign')
    generic = spec.to_bayesian_optimization_input()

    assert generic.upload_id == 'u'
    assert generic.user_id == 'x'
    assert generic.schema_name == SCHEMA_NAME
    assert generic.batch_size == 3
    assert generic.campaign_name == 'my_campaign'

    assert {
        variable.name: (variable.lower_bound, variable.upper_bound)
        for variable in generic.variables
    } == BOUNDS
    assert {
        variable.name: variable.decimals for variable in generic.variables
    } == DECIMALS
    assert all(variable.kind == 'continuous' for variable in generic.variables)

    [target] = generic.targets
    assert target.name == 'film_thickness'
    assert target.mode == 'MATCH'
    assert target.match_value == 300.0
    assert target.sigma == pytest.approx(30.0)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'target_film_thickness': 0.0},
        {'target_film_thickness': -10.0},
        {'batch_size': 0},
    ],
)
def test_invalid_input(kwargs):
    with pytest.raises(ValidationError):
        SpinCoatingOptimizationInput(
            **{'upload_id': 'u', 'user_id': 'x', 'target_film_thickness': 300.0}
            | kwargs
        )


def test_input_schema():
    """The action form only asks for the upload, target, steps and campaign name."""
    pytest.importorskip('temporalio')
    from nomad.actions.manager import _get_param_schema

    from nomad_bayesian_optimization.actions.spin_coating.workflows import (
        SpinCoatingOptimizationWorkflow,
    )

    schema = _get_param_schema(SpinCoatingOptimizationWorkflow.run)
    assert set(schema['properties']) == {
        'upload_id',
        'target_film_thickness',
        'batch_size',
        'campaign_name',
    }


def test_workflow_inherits_signals():
    """The subclassed workflow is a valid Temporal workflow with the base signals."""
    workflow = pytest.importorskip('temporalio.workflow')

    from nomad_bayesian_optimization.actions.spin_coating.workflows import (
        SpinCoatingOptimizationWorkflow,
    )

    definition = workflow._Definition.must_from_class(SpinCoatingOptimizationWorkflow)
    assert definition.name == 'SpinCoatingOptimizationWorkflow'
    assert set(definition.signals) == {'decision'}


def test_decision_signal_schema():
    """The decision form asks for the action and the measured entry IDs at once."""
    pytest.importorskip('temporalio')
    from nomad.actions.manager import _get_signal_schema

    from nomad_bayesian_optimization.actions.campaign.models import DecisionInput
    from nomad_bayesian_optimization.actions.spin_coating.workflows import (
        SpinCoatingOptimizationWorkflow,
    )

    decision = DecisionInput()
    assert decision.action == 'accept'
    assert decision.entry_ids == []

    schema = _get_signal_schema(SpinCoatingOptimizationWorkflow.decision)
    assert schema['properties']['action']['enum'] == ['accept', 'reject', 'finish']
    assert schema['properties']['entry_ids']['type'] == 'array'
    assert schema['properties']['entry_ids']['items'] == {'type': 'string'}
    assert not schema.get('required')


def test_recommendations_are_rounded():
    """Suggestions lie within the bounds and are rounded to the lab resolution,
    both in the returned records and in the persisted campaign."""
    pytest.importorskip('baybe')
    pytest.importorskip('temporalio')
    import asyncio
    import json

    from nomad_bayesian_optimization.actions.campaign.activities import (
        recommend_next,
    )
    from nomad_bayesian_optimization.actions.campaign.models import RecommendInput
    from nomad_bayesian_optimization.campaign_builder import (
        build_campaign,
        seed_campaign,
    )
    from nomad_bayesian_optimization.campaign_converter import recommended_records

    campaign = build_campaign(_input().to_bayesian_optimization_input())
    records = [
        {
            'precursor_concentration': 1.2,
            'spin_coating_speed': 4000.0,
            'spinning_time': 30.0,
            'film_thickness': 400.0,
        },
        {
            'precursor_concentration': 0.5,
            'spin_coating_speed': 2000.0,
            'spinning_time': 20.0,
            'film_thickness': 190.0,
        },
    ]
    assert seed_campaign(campaign, records) == 2

    output = asyncio.run(
        recommend_next(
            RecommendInput(
                campaign_json=campaign.to_json(), batch_size=2, decimals=DECIMALS
            )
        )
    )

    assert len(output.records) == 2
    for record in output.records:
        for name, (lower, upper) in BOUNDS.items():
            assert lower <= record[name] <= upper
            assert record[name] == round(record[name], DECIMALS[name])
    assert recommended_records(json.loads(output.campaign_json)) == output.records


def test_clear_recommendation():
    """Finishing drops the pending recommendation from the campaign while keeping
    the recorded measurements and their entry IDs."""
    pytest.importorskip('baybe')
    pytest.importorskip('temporalio')
    import asyncio
    import json

    from nomad_bayesian_optimization.actions.campaign.activities import (
        clear_recommendation,
        recommend_next,
    )
    from nomad_bayesian_optimization.actions.campaign.models import RecommendInput
    from nomad_bayesian_optimization.campaign_builder import (
        build_campaign,
        seed_campaign,
    )
    from nomad_bayesian_optimization.campaign_converter import (
        measured_records,
        recommended_records,
    )

    campaign = build_campaign(_input().to_bayesian_optimization_input())
    seed_campaign(
        campaign,
        [
            {
                'precursor_concentration': 1.2,
                'spin_coating_speed': 4000.0,
                'spinning_time': 30.0,
                'film_thickness': 400.0,
                'entry_id': 'e1',
            }
        ],
    )
    output = asyncio.run(
        recommend_next(RecommendInput(campaign_json=campaign.to_json()))
    )
    assert recommended_records(json.loads(output.campaign_json))

    cleared = json.loads(asyncio.run(clear_recommendation(output.campaign_json)))
    assert recommended_records(cleared) == []
    assert [record['entry_id'] for record in measured_records(cleared)] == ['e1']
