from nomad.config.models.plugins import ExampleUploadEntryPoint
from nomad.utils import strip

getting_started = ExampleUploadEntryPoint(
    title='Getting started with Bayesian optimization',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    path='example_uploads/getting_started',
    category='Lab automation',
)
