from nomad.config.models.plugins import ExampleUploadEntryPoint, UploadResource
from nomad.utils import strip

getting_started = ExampleUploadEntryPoint(
    title='Getting started',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources='example_uploads/getting_started/*',
    category='Bayesian optimization',
)

# optimization_tasks = ExampleUploadEntryPoint(
#     title='Example optimization tasks',
#     description=strip("""
#     Contains a series of optimization runs for exploring what bayesian optimization data
# 	looks like.
#     """),
#     path='example_uploads/optimization_tasks',
#     category='Bayesian optimization',
# )

file = ExampleUploadEntryPoint(
    title='file',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources='example_uploads/getting_started/README.md',
    category='Lab automation',
)

file_target = ExampleUploadEntryPoint(
    title='file_target',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources=UploadResource(
        path='example_uploads/getting_started/README.md', target='folder_1/folder2/'
    ),
    category='Lab automation',
)

folder = ExampleUploadEntryPoint(
    title='folder',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources='example_uploads/getting_started',
    category='Lab automation',
)

folder_contents = ExampleUploadEntryPoint(
    title='folder_contents',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources='example_uploads/getting_started/*',
    category='Lab automation',
)

zip_unpack = ExampleUploadEntryPoint(
    title='zip_unpack',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources='example_uploads/getting_started/nomad-oasis.zip',
    category='Lab automation',
)

multiple_files = ExampleUploadEntryPoint(
    title='multiple_files',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources=[
        'example_uploads/getting_started/notebook.ipynb',
        'example_uploads/getting_started/nomad-oasis.zip',
    ],
    category='Lab automation',
)

zip_online_unpack = ExampleUploadEntryPoint(
    title='zip_online_unpack',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources='https://nomad-lab.eu/prod/v1/docs/assets/nomad-oasis.zip',
    category='Lab automation',
)

multiple_online_files = ExampleUploadEntryPoint(
    title='multiple_online_files',
    description=strip("""
    Example upload that contains a Jupyter notebook demonstrating the basics of
    Bayesian optimization.
    """),
    resources=[
        'https://nomad-lab.eu/prod/v1/docs/assets/nomad-oasis.zip',
        'https://nomad-lab.eu/prod/v1/docs/howto/oasis/install.html',
    ],
    category='Lab automation',
)
