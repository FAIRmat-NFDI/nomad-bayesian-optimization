import pandas as pd
import plotly.graph_objects as go
from baybe.serialization.utils import deserialize_dataframe
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.metainfo import JSON, MEnum, MSection, Quantity, SchemaPackage, SubSection

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
            'SequentialGreedyRecommender',
            'RandomRecommender',
        )
    )
    surrogate_model = SubSection(section_def=SurrogateModel)
    initial_recommender = SubSection(section_def='Recommender')
    recommender = SubSection(section_def='Recommender')
    acquisition_function = SubSection(section_def=AcquisitionFunction)
    hybrid_sampler = Quantity(type=MEnum('Farthest'))
    sampling_percentage = Quantity(type=float)


class Step(MSection):
    type = Quantity(type=MEnum('TwoPhaseMetaRecommender'))


class BayesianOptimization(PlotSection, Schema):
    """Represents a single Bayesian optimization task."""

    searchspace = SubSection(section_def=SearchSpace)
    objective = SubSection(section_def=Objective)
    recommender = SubSection(section_def=Recommender)
    entries = Quantity(
        type=str,
        shape=['*'],
        description='List of entries connected to the optimization.',
    )
    status = Quantity(
        type=MEnum('INITIALIZING', 'SUGGESTING', 'ACQUIRING', 'FINISHED', 'ERROR'),
        default='INITIALIZING',
    )
    baybe_campaign = Quantity(
        type=JSON,
        description='Contains the fully serialized BayBE campaign that represents this Bayesian Optimization.',
    )

    def from_baybe_campaign(campaign):
        dictionary = campaign.to_dict()
        result = BayesianOptimization.m_from_dict(dictionary)
        result.baybe_campaign = dictionary
        return result

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # If this entry has been created from a BayBE run, we use that data to
        # create plots.
        if self.baybe_campaign:
            df = deserialize_dataframe(self.baybe_campaign['_cached_recommendation'])
            print(df)
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
                # autosize=,
            )

            self.figures = [
                PlotlyFigure(label='Progress', figure=figure1.to_plotly_json()),
                PlotlyFigure(label='Steps', figure=figure2.to_plotly_json()),
            ]


m_package.__init_metainfo__()
