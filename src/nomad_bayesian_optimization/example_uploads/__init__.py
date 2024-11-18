from nomad.config.models.plugins import ExampleUploadEntryPoint
from nomad.utils import strip

getting_started = ExampleUploadEntryPoint(
    title='Getting started',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    path='example_uploads/getting_started/*',
    category='Bayesian optimization',
)

optimization_tasks = ExampleUploadEntryPoint(
    title='Example optimization tasks',
    description=strip("""
    Contains a series of optimization runs for exploring what bayesian optimization data
	looks like.
    """),
    path='example_uploads/optimization_tasks/*',
    category='Bayesian optimization',
)
