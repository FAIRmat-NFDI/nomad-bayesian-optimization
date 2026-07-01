import os.path

from nomad.client import normalize_all, parse


def test_schema():
    """The BayesianOptimization schema parses and normalizes an archive.

    This exercises the schema and its ``normalize`` method (n_steps and the
    generated figures) without requiring BayBE, complementing the parser tests.
    """
    n_steps_in_fixture = 3

    test_file = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'bayesian_optimization.archive.yaml'
    )
    archive = parse(test_file)[0]
    normalize_all(archive)
    data = archive.data

    assert data.status == 'Finished'
    assert data.n_steps == n_steps_in_fixture
    assert len(data.steps) == n_steps_in_fixture
    assert {p.name for p in data.parameters} == {'pressure', 'substrate'}
    assert data.objective.type == 'SingleTargetObjective'
    assert data.objective.targets[0].name == 'yield'

    # normalize() creates one progress figure per target plus a search-space table.
    assert len(data.figures) == len(data.objective.targets) + 1
