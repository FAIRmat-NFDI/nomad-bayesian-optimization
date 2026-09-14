from nomad.client import normalize_all
from nomad.datamodel import EntryArchive, EntryMetadata

from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
    Objective,
    Target,
)
from nomad_bayesian_optimization.step_schema import (
    attach_step_package,
    derive_step_fields,
    make_step_instance,
)


def _build_archive():
    """Build a ``BayesianOptimization`` archive with a generated, typed step schema.

    Mirrors what the parser does, but without BayBE: it derives the step fields
    from a small campaign dict (plus injected field metadata), attaches the
    generated ``CampaignStep`` schema to ``archive.definitions`` and appends a few
    typed step instances.
    """
    campaign = {
        'searchspace': {
            'discrete': {
                'parameters': [
                    {'type': 'NumericalDiscreteParameter', 'name': 'pressure'},
                    {'type': 'CategoricalParameter', 'name': 'substrate'},
                ]
            },
            'continuous': {'parameters': []},
        },
        'objective': {'type': 'SingleTargetObjective', 'target': {'name': 'yield'}},
    }
    field_meta = {
        'pressure': {
            'type': 'float',
            'unit': 'pascal',
            'description': 'Chamber pressure',
        },
        'yield': {'type': 'float', 'unit': None, 'description': 'Reaction yield'},
    }
    field_specs = derive_step_fields(campaign, field_meta)

    archive = EntryArchive(metadata=EntryMetadata())
    archive.data = BayesianOptimization(status='Finished')
    objective = Objective(type='SingleTargetObjective')
    objective.targets.append(Target(name='yield', minimize=False))
    archive.data.objective = objective

    step_def = attach_step_package(archive, field_specs)
    rows = [
        {'pressure': 1.0, 'substrate': 'Si', 'yield': 60.0},
        {'pressure': 2.0, 'substrate': 'SiC', 'yield': 75.0},
        {'pressure': 3.0, 'substrate': 'GaN', 'yield': 82.0},
    ]
    for row in rows:
        archive.data.steps.append(
            make_step_instance(step_def, row, field_specs, recommended=False)
        )
    return archive, field_specs


def test_schema():
    """The BayesianOptimization schema normalizes an archive with typed steps.

    Exercises the generated ``CampaignStep`` schema and ``normalize`` (n_steps and
    the generated figures) without requiring BayBE, complementing the parser tests.
    """
    n_steps_in_fixture = 3

    archive, _ = _build_archive()
    normalize_all(archive)
    data = archive.data

    assert data.status == 'Finished'
    assert data.n_steps == n_steps_in_fixture
    assert len(data.steps) == n_steps_in_fixture
    assert data.objective.type == 'SingleTargetObjective'
    assert data.objective.targets[0].name == 'yield'

    # The generated step schema carries the field types, units and descriptions.
    quantities = {
        q.name: q for q in archive.definitions.section_definitions[0].quantities
    }
    assert set(quantities) == {'pressure', 'substrate', 'yield'}
    assert str(quantities['pressure'].unit) == 'pascal'
    assert quantities['pressure'].description == 'Chamber pressure'
    assert quantities['substrate'].type.standard_type() == 'str'

    # Step values are stored on the typed instances.
    assert [float(getattr(s, 'yield')) for s in data.steps] == [60.0, 75.0, 82.0]

    # normalize() creates one progress figure per target plus a search-space table.
    assert len(data.figures) == len(data.objective.targets) + 1
