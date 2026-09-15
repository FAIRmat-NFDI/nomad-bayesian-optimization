import pytest
from nomad.client import normalize_all
from nomad.datamodel import EntryArchive, EntryMetadata

from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
    Objective,
    Target,
    _best_so_far,
)
from nomad_bayesian_optimization.step_schema import (
    attach_step_package,
    derive_step_fields,
    make_step_instance,
)


def _build_archive(target_unit: str | None = None):
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
        'yield': {
            'type': 'float',
            'unit': target_unit,
            'description': 'Reaction yield',
        },
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

    # normalize() creates one progress figure per target.
    assert len(data.figures) == len(data.objective.targets)
    figure = data.figures[0]
    assert figure.label == 'yield'
    measured, best = figure.figure['data']
    assert list(measured['x']) == [1, 2, 3]
    assert list(measured['y']) == [60.0, 75.0, 82.0]
    assert list(best['x']) == [1, 2, 3]
    assert list(best['y']) == [60.0, 75.0, 82.0]
    assert figure.figure['layout']['xaxis']['title']['text'] == 'Step'
    assert figure.figure['layout']['yaxis']['title']['text'] == 'yield'


def test_progress_figure_skips_recommended_steps():
    """Pending recommendations have no measured target value and are not plotted."""
    archive, field_specs = _build_archive(target_unit='kelvin')
    step_def = archive.definitions.section_definitions[0]
    archive.data.steps.append(
        make_step_instance(
            step_def,
            {'pressure': 4.0, 'substrate': 'Si'},
            field_specs,
            recommended=True,
        )
    )
    normalize_all(archive)

    measured, best = archive.data.figures[0].figure['data']
    assert list(measured['x']) == [1, 2, 3]
    assert list(best['x']) == [1, 2, 3]
    layout = archive.data.figures[0].figure['layout']
    assert layout['yaxis']['title']['text'] == 'yield (K)'


@pytest.mark.parametrize(
    'target, values, expected',
    [
        pytest.param(
            Target(minimize=False), [70.0, 60.0, 80.0], [70.0, 70.0, 80.0], id='max'
        ),
        pytest.param(
            Target(minimize=True), [12.0, 8.0, 10.0], [12.0, 8.0, 8.0], id='min'
        ),
        pytest.param(
            Target(
                minimize=False,
                transformation_parameters={
                    'type': 'BellTransformation',
                    'center': 80.0,
                    'sigma': 5.0,
                },
            ),
            [70.0, 95.0, 78.0],
            [70.0, 70.0, 78.0],
            id='bell',
        ),
        pytest.param(
            Target(
                minimize=False,
                transformation_parameters={
                    'type': 'TriangularTransformation',
                    'cutoffs': {'lower': 1.5, 'upper': 2.5},
                    'peak': 2.0,
                },
            ),
            [1.6, 2.2, 1.9],
            [1.6, 2.2, 1.9],
            id='triangular',
        ),
    ],
)
def test_best_so_far(target, values, expected):
    """The running best depends on the target's direction or match value."""
    assert _best_so_far(values, target) == expected
