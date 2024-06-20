import plotly.express as px
from baybe.serialization.utils import deserialize_dataframe
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.metainfo import MEnum, MSection, Quantity, SchemaPackage, SubSection
from plotly.subplots import make_subplots

m_package = SchemaPackage()


class Bounds(MSection):
    lower = Quantity(type=MEnum('CategoricalParameter', 'NumericalContinuousParameter'))
    upper = Quantity(type=str)


class Parameter(MSection):
    """Baysian optimization run."""

    type = Quantity(type=MEnum('CategoricalParameter', 'NumericalContinuousParameter'))
    name = Quantity(type=str)
    values = Quantity(type=str, shape=['*'])
    bounds = SubSection(section_def=Bounds)
    encoding = Quantity(type=str)


class Discrete(MSection):
    parameters = SubSection(section_def=Parameter, repeats=True)


class Continuous(MSection):
    parameters = SubSection(section_def=Parameter, repeats=True)


class SearchSpace(MSection):
    discrete = SubSection(section_def=Discrete)
    continuous = SubSection(section_def=Continuous)


class Target(MSection):
    type = Quantity(type=MEnum('NumericalTarget'))
    name = Quantity(type=str)
    mode = Quantity(type=MEnum('MATCH'))
    transformation = Quantity(type=MEnum('BELL'))
    bounds = SubSection(section_def=Bounds)


class Objective(MSection):
    type = Quantity(type=MEnum('SingleTargetObjective'))
    target = SubSection(section_def=Target)


class Recommender(MSection):
    type = Quantity(type=MEnum('TwoPhaseMetaRecommender'))


class Step(MSection):
    type = Quantity(type=MEnum('TwoPhaseMetaRecommender'))


class BayesianOptimization(PlotSection, Schema):
    """Baysian optimization run."""

    searchspace = SubSection(section_def=SearchSpace)
    objective = SubSection(section_def=Objective)
    recommender = SubSection(section_def=Recommender)
    optimization = Quantity(type=str, description='base64 encoded optimization run')
    entries = Quantity(
        type=str,
        shape=['*'],
        description='List of entries connected to the optimization.',
    )

    def from_baybe_campaign(campaign):
        dictionary = campaign.to_dict()
        result = BayesianOptimization.m_from_dict(dictionary)
        result.optimization = dictionary['_measurements_exp']
        return result

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        df = deserialize_dataframe(self.optimization)
        target_name = self.objective.target.name
        first_line = px.scatter(
            df,
            x='BatchNr',
            y=target_name,
            title='Optimization result',
            labels={
                'target_name': 'Refractive index',
                'BatchNr': 'Iteration',
            },
        )
        figure1 = make_subplots(rows=1, cols=1, shared_yaxes=True)
        figure1.add_trace(first_line.data[0], row=1, col=1)
        self.figures = [
            PlotlyFigure(label='Optimization results', figure=figure1.to_plotly_json())
        ]


m_package.__init_metainfo__()
