import pandas as pd
import plotly.graph_objects as go
from baybe.serialization.utils import deserialize_dataframe
from nomad.datamodel.data import ArchiveSection, Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.metainfo import (
    JSON,
    MEnum,
    MSection,
    Quantity,
    SchemaPackage,
    Section,
    SubSection,
)

m_package = SchemaPackage()


class Parameter(MSection):
    """Parameter."""

    name = Quantity(type=str)


class CategoricalParameter(Parameter):
    """Discrete categorical parameter."""

    type = Quantity(type=MEnum('CategoricalParameter'))
    values = Quantity(type=str, shape=['*'])
    encoding = Quantity(type=str)


class Bounds(MSection):
    lower = Quantity(type=float)
    upper = Quantity(type=float)


class NumericalContinuousParameter(Parameter):
    """Continuous numerical parameter."""

    type = Quantity(type=MEnum('NumericalContinuousParameter'))
    bounds = SubSection(section_def=Bounds)


class Discrete(MSection):
    """Container for a list of discrete parameters in a search space."""

    parameters = SubSection(section_def=CategoricalParameter, repeats=True)


class Continuous(MSection):
    """Container for a list of continuous parameters in a search space."""

    parameters = SubSection(section_def=NumericalContinuousParameter, repeats=True)


class SearchSpace(MSection):
    """The search space of the task. Divided into two categories: discrete and
    continuous."""

    discrete = SubSection(section_def=Discrete)
    continuous = SubSection(section_def=Continuous)


class Target(MSection):
    """The target specification for the optimization."""

    type = Quantity(type=MEnum('NumericalTarget'))
    name = Quantity(type=str)
    mode = Quantity(type=MEnum('MATCH'))
    transformation = Quantity(type=MEnum('BELL'))
    bounds = SubSection(section_def=Bounds)


class Objective(MSection):
    type = Quantity(type=MEnum('SingleTargetObjective'))
    target = SubSection(section_def=Target)


class KernelFactory(MSection):
    type = Quantity(type=MEnum('DefaultKernelFactory'))


class SurrogateModel(MSection):
    type = Quantity(type=MEnum('GaussianProcessSurrogate'))
    kernel_factor = SubSection(section_def=KernelFactory)


class AcquisitionFunction(MSection):
    type = Quantity(type=MEnum('qExpectedImprovement'))


class Recommender(MSection):
    type = Quantity(
        type=MEnum(
            'TwoPhaseMetaRecommender',
            'NaiveHybridSpaceRecommender',
            'RandomRecommender',
        )
    )
    surrogate_model = SubSection(section_def=SurrogateModel)
    initial_recommender = SubSection(section_def='Recommender')
    recommender = SubSection(section_def='Recommender')
    acquisition_function = SubSection(section_def=AcquisitionFunction)
    hybrid_sampler = Quantity(type=MEnum('Farthest'))
    sampling_percentage = Quantity(type=float)


class Optimization(MSection):
    """Contains information about the optimization procedure."""

    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )

    finished = Quantity(
        type=bool,
        description='Is the optimization finished',
        a_eln=ELNAnnotation(
            component='BoolEditQuantity',
        ),
    )
    status = Quantity(
        type=MEnum('Initializing', 'Suggesting', 'Acquiring', 'Finished', 'Error'),
        default='Initializing',
        description='Optimization status.',
        a_eln=ELNAnnotation(
            component='EnumEditQuantity',
        ),
    )
    n_steps = Quantity(type=int, description='Number of steps in optimization.')
    entries = Quantity(
        type=str,
        shape=['*'],
        description='List of entry_ids connected to the optimization.',
    )


class Material(MSection):
    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )
    substance_name = Quantity(
        type=str,
        a_eln=ELNAnnotation(
            component='StringEditQuantity',
        ),
    )


class Source(MSection):
    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )
    materials = SubSection(section_def=Material, repeats=True)


class Step(MSection):
    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )
    sources = SubSection(section_def=Source, repeats=True)


class MySection(ArchiveSection):
    my_quantity = Quantity(type=str)


class BayesianOptimization(PlotSection, Schema):
    """Represents a single Bayesian optimization task."""

    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )

    searchspace = SubSection(section_def=SearchSpace)
    objective = SubSection(section_def=Objective)
    recommender = SubSection(section_def=Recommender)
    optimization = SubSection(section_def=Optimization)
    steps = SubSection(section_def=Step, repeats=True)
    baybe_campaign = Quantity(
        type=JSON,
        description="""
        Contains the fully serialized BayBE campaign that represents this Bayesian
        Optimization.
        """,
    )
    my_subsection = SubSection(section_def=MySection)

    def from_baybe_campaign(campaign):
        # Populate the parts that are directly compatible with a BayBE campaign
        # data model
        dictionary = campaign.to_dict()
        result = BayesianOptimization.m_from_dict(dictionary)
        result.baybe_campaign = dictionary

        # Populate additional parts that cannot be directly read from a BayBE
        # campaign
        df = deserialize_dataframe(dictionary['_measurements_exp'])
        result.optimization = Optimization(n_steps=df.shape[0], status='Finished')

        return result

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        self.my_subsection = MySection(my_quantity='Test')
        self.reference_direct = archive.data.my_subsection
        self.reference_inherited = archive.data.my_subsection

        # If this entry has been created from a BayBE run, we use that data to
        # create plots.
        if self.baybe_campaign:
            # BayBE encodes the optimization progress in a special way that requires
            # it to be read with a utility function.
            df = deserialize_dataframe(self.baybe_campaign['_measurements_exp'])

            # Generate a plot that shows how the optimization progresses each step
            target_name = self.objective.target.name
            figure1 = go.Figure()
            figure1.add_trace(
                go.Scatter(
                    x=df['BatchNr'],
                    y=df[target_name],
                    mode='lines+markers',
                )
            )
            figure1.update_layout(
                template='plotly_white',
                title='Progress',
                xaxis_title='Iteration',
                yaxis_title=target_name,
            )

            # Create a table of the traversed search space from last to first
            # step. If there is a non-tried recommendation, add it to the table
            # as a new column
            cached_recommendation = deserialize_dataframe(
                self.baybe_campaign['_cached_recommendation']
            )
            if not cached_recommendation.empty:
                df = pd.concat(
                    [df, cached_recommendation], ignore_index=True, sort=False
                )
            df = df[::-1]

            figure2 = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=list(df.columns),
                            align='left',
                        ),
                        cells=dict(
                            values=df.transpose().values.tolist(),
                            align='left',
                        ),
                    )
                ]
            )
            figure2.update_layout(
                template='plotly_white',
                margin=dict(l=0, r=0, t=0, b=0),
                width=800,
            )

            self.figures = [
                PlotlyFigure(label='Progress', figure=figure1.to_plotly_json()),
                PlotlyFigure(label='Steps', figure=figure2.to_plotly_json()),
            ]


m_package.__init_metainfo__()
