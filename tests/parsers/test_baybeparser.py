"""Tests for the BayBE campaign parser.

The parser consumes serialized BayBE campaigns (``campaign.to_json()``). Rather
than committing serialized campaigns (which embed version-sensitive, pickled
dataframes), the numerical campaign types are built and serialized on the fly so
the fixtures always match the installed BayBE version. The substance campaign is
tested from a small, faithful static fixture (``tests/data/campaign_substance.json``)
so the test suite does not require the heavy optional ``baybe[chem]`` stack.

See the NOMAD plugin testing guide:
https://fairmat-nfdi.github.io/nomad-docs/howto/plugins/plugins.html#testing
"""

import os.path

import pandas as pd
import pytest
from nomad.client import normalize_all, parse

from nomad_bayesian_optimization.naming import sanitize_quantity_name

# BayBE is required to build the serialized campaign fixtures.
pytest.importorskip('baybe')

from baybe import Campaign  # noqa: E402
from baybe.objectives import (  # noqa: E402
    DesirabilityObjective,
    SingleTargetObjective,
)
from baybe.parameters import (  # noqa: E402
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace  # noqa: E402
from baybe.targets import NumericalTarget  # noqa: E402


def _serialize(campaign: Campaign, tmp_path, name: str) -> str:
    """Serialize a campaign to ``<tmp_path>/<name>.json`` and return the path."""
    path = tmp_path / f'{name}.json'
    path.write_text(campaign.to_json())
    return str(path)


def build_discrete(tmp_path):
    """Single numerical target over purely numerical-discrete parameters."""
    parameters = [
        NumericalDiscreteParameter(name='pressure', values=[1.0, 2.0, 3.0]),
        NumericalDiscreteParameter(name='time', values=[10.0, 20.0, 30.0]),
    ]
    searchspace = SearchSpace.from_product(parameters)
    target = NumericalTarget.match_bell(name='yield', match_value=80.0, sigma=5.0)
    campaign = Campaign(searchspace, SingleTargetObjective(target=target))
    campaign.add_measurements(
        pd.DataFrame(
            {
                'pressure': [1.0, 2.0, 3.0],
                'time': [10.0, 20.0, 30.0],
                'yield': [60.0, 75.0, 82.0],
            }
        )
    )
    path = _serialize(campaign, tmp_path, 'discrete')
    return path, {
        'param_names': {'pressure', 'time'},
        'param_types': {'NumericalDiscreteParameter'},
        'objective_type': 'SingleTargetObjective',
        'target_names': ['yield'],
        'minimize': [False],
        'n_steps': 3,
        'measured_target_value': 82.0,
    }


def build_continuous(tmp_path):
    """Single numerical target (minimized) over continuous parameters."""
    parameters = [
        NumericalContinuousParameter(name='temperature', bounds=(300, 600)),
        NumericalContinuousParameter(name='flow', bounds=(0.2, 5.0)),
    ]
    searchspace = SearchSpace.from_product(parameters)
    target = NumericalTarget(name='cost', minimize=True)
    campaign = Campaign(searchspace, SingleTargetObjective(target=target))
    campaign.add_measurements(
        pd.DataFrame(
            {
                'temperature': [350.0, 500.0],
                'flow': [1.0, 3.0],
                'cost': [12.0, 8.0],
            }
        )
    )
    path = _serialize(campaign, tmp_path, 'continuous')
    return path, {
        'param_names': {'temperature', 'flow'},
        'param_types': {'ContinuousParameter'},
        'objective_type': 'SingleTargetObjective',
        'target_names': ['cost'],
        'minimize': [True],
        'n_steps': 2,
        'measured_target_value': 8.0,
    }


def build_hybrid(tmp_path):
    """Single target over a mix of categorical and continuous parameters."""
    parameters = [
        CategoricalParameter(
            name='substrate', values=['Si', 'SiC', 'GaN'], encoding='OHE'
        ),
        NumericalContinuousParameter(name='temperature', bounds=(300, 600)),
    ]
    searchspace = SearchSpace.from_product(parameters)
    target = NumericalTarget.match_bell(
        name='refractive_index', match_value=2.0, sigma=0.2
    )
    campaign = Campaign(searchspace, SingleTargetObjective(target=target))
    campaign.add_measurements(
        pd.DataFrame(
            {
                'substrate': ['Si', 'SiC'],
                'temperature': [400.0, 450.0],
                'refractive_index': [1.9, 2.05],
            }
        )
    )
    path = _serialize(campaign, tmp_path, 'hybrid')
    return path, {
        'param_names': {'substrate', 'temperature'},
        'param_types': {'CategoricalParameter', 'ContinuousParameter'},
        'objective_type': 'SingleTargetObjective',
        'target_names': ['refractive_index'],
        'minimize': [False],
        'n_steps': 2,
        'measured_target_value': 2.05,
    }


def build_desirability(tmp_path):
    """Multi-target desirability objective with weights."""
    parameters = [
        NumericalDiscreteParameter(name='pressure', values=[1.0, 2.0, 3.0]),
        NumericalContinuousParameter(name='temperature', bounds=(300, 600)),
    ]
    searchspace = SearchSpace.from_product(parameters)
    targets = [
        NumericalTarget.match_bell(name='yield', match_value=80.0, sigma=5.0),
        NumericalTarget.match_bell(name='purity', match_value=95.0, sigma=3.0),
    ]
    objective = DesirabilityObjective(
        targets=targets, weights=[2.0, 1.0], scalarizer='MEAN'
    )
    campaign = Campaign(searchspace, objective)
    campaign.add_measurements(
        pd.DataFrame(
            {
                'pressure': [1.0, 2.0],
                'temperature': [400.0, 450.0],
                'yield': [70.0, 78.0],
                'purity': [90.0, 96.0],
            }
        )
    )
    path = _serialize(campaign, tmp_path, 'desirability')
    return path, {
        'param_names': {'pressure', 'temperature'},
        'param_types': {'NumericalDiscreteParameter', 'ContinuousParameter'},
        'objective_type': 'DesirabilityObjective',
        'target_names': ['yield', 'purity'],
        'minimize': [False, False],
        'weights': [2.0, 1.0],
        'scalarizer': 'MEAN',
        'n_steps': 2,
        'measured_target_value': 78.0,
    }


CAMPAIGN_BUILDERS = {
    'discrete': build_discrete,
    'continuous': build_continuous,
    'hybrid': build_hybrid,
    'desirability': build_desirability,
}


@pytest.mark.parametrize('builder_name', list(CAMPAIGN_BUILDERS))
def test_parse_campaign(builder_name, tmp_path):
    """Each common campaign type parses and normalizes into the schema."""
    path, expected = CAMPAIGN_BUILDERS[builder_name](tmp_path)

    archive = parse(path)[0]
    normalize_all(archive)
    data = archive.data

    # Parameters
    assert {p.name for p in data.parameters} == expected['param_names']
    assert {p.m_def.name for p in data.parameters} == expected['param_types']

    # Objective and targets
    assert data.objective.type == expected['objective_type']
    assert [t.name for t in data.objective.targets] == expected['target_names']
    assert [bool(t.minimize) for t in data.objective.targets] == expected['minimize']

    # The generated per-campaign step schema is stored in the archive
    # definitions as a subclass of ``Step`` with one quantity per parameter/target.
    assert archive.definitions is not None
    section_defs = archive.definitions.section_definitions
    assert len(section_defs) == 1
    step_section = section_defs[0]
    assert step_section.name == 'CampaignStep'
    assert any(base.name == 'Step' for base in step_section.base_sections)
    quantity_names = {q.name for q in step_section.quantities}
    expected_quantities = set(expected['param_names']) | {
        sanitize_quantity_name(name) for name in expected['target_names']
    }
    assert quantity_names == expected_quantities

    # Steps / measurements are stored as typed instances of the generated schema.
    assert data.n_steps == expected['n_steps']
    assert len(data.steps) == expected['n_steps']
    target_quantity = sanitize_quantity_name(expected['target_names'][0])
    measured = [
        getattr(step, target_quantity)
        for step in data.steps
        if getattr(step, target_quantity, None) is not None
    ]
    assert expected['measured_target_value'] in measured

    # A progress figure is produced for each target, plus a search-space table.
    assert len(data.figures) == len(expected['target_names']) + 1

    if 'weights' in expected:
        assert [t.weight for t in data.objective.targets] == expected['weights']
        assert data.objective.scalarizer == expected['scalarizer']


def test_parse_substance_campaign():
    """A campaign with a SubstanceParameter is parsed from a static fixture."""
    test_file = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'campaign_substance.json'
    )
    archive = parse(test_file)[0]
    normalize_all(archive)
    data = archive.data

    assert len(data.parameters) == 1
    substance = data.parameters[0]
    assert substance.m_def.name == 'SubstanceParameter'
    assert substance.name == 'solvent'
    assert {s.name for s in substance.values} == {'water', 'methanol', 'ethanol'}
    assert {s.smiles for s in substance.values} == {'O', 'CO', 'CCO'}
    assert substance.encoding == 'ECFP'

    assert data.objective.type == 'SingleTargetObjective'
    assert data.objective.targets[0].name == 'solubility'
