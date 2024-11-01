def test_importing_app():
    # this will raise an exception if pydantic model validation fails for the app
    from nomad_bayesian_optimization.apps import (
        bayesian_optimization_tasks,  # noqa: F401
    )
