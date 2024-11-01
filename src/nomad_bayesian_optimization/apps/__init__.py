from nomad.config.models.plugins import AppEntryPoint

from .tasks import app

bayesian_optimization_tasks = AppEntryPoint(
    name='Bayesian Optimizations Tasks',
    description='App for Bayesian Optimization Tasks',
    app=app,
)
