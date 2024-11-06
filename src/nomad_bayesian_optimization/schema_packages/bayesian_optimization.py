import plotly.graph_objects as go
from baybe.serialization.utils import deserialize_dataframe
from nomad.datamodel.data import ArchiveSection, Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
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


class Parameter(ArchiveSection):
    """Parameter."""

    name = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    value_reference = Quantity(
        type=Quantity,
        a_eln=ELNAnnotation(component=ELNComponentEnum.ReferenceEditQuantity),
    )
    definition = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )  # nomad_bayesian_optimization.schema_packages.experiments.CVDExperiment.gas_flow_rate


class ContinuousParameter(Parameter):
    """Continuous parameter."""

    lower_bound = Quantity(
        type=float,
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )
    upper_bound = Quantity(
        type=float,
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )


class DiscreteParameter(Parameter):
    """Discrete parameter."""

    pass


class NumericalDiscreteParameter(DiscreteParameter):
    values = Quantity(
        type=float,
        shape=['*'],
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )  # TODO populate values from m_def


class CategoricalParameter(DiscreteParameter):
    values = Quantity(
        type=str,
        shape=['*'],
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )  # TODO populate values from m_def


class BoSubstance(ArchiveSection):
    name = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    smiles = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )


class SubstanceParameter(DiscreteParameter):
    values = SubSection(
        section_def=BoSubstance,
        repeats=True,
    )


class Bounds(MSection):
    lower = Quantity(type=float)
    upper = Quantity(type=float)


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


class Step(MSection):
    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )
    entry = Quantity(
        type=Schema,
        a_eln=dict(component='ReferenceEditQuantity'),
    )
    value_used = Quantity(
        type=JSON, desription='The final values passed to the optimization procedure.'
    )
    value_recommended = Quantity(
        type=JSON,
        desription='The values suggested by the Bayesian optimization procedure.',
    )

    def normalize(self, archive, logger) -> None:
        # TODO: Extract value_final from the entry reference using the search
        # space information
        if not self.value_used and self.entry:
            pass


class Optimization(ArchiveSection):
    """Contains information about the optimization procedure."""

    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )
    status = Quantity(
        type=MEnum('Initializing', 'Suggesting', 'Acquiring', 'Finished', 'Error'),
        default='Initializing',
        description='Optimization status.',
    )
    steps = SubSection(section_def=Step, repeats=True)
    n_steps = Quantity(
        type=int,
        description='Number of steps in optimization.',
    )

    def normalize(self, archive, logger) -> None:
        self.n_steps = len(self.steps or [])


class BayesianOptimization(PlotSection, Schema):
    """Represents a single Bayesian optimization task."""

    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width='600px',
        )
    )

    # searchspace = SubSection(section_def=SearchSpace)
    parameters = SubSection(section_def=Parameter, repeats=True)
    objective = SubSection(section_def=Objective)
    recommender = SubSection(section_def=Recommender)
    optimization = SubSection(section_def=Optimization)
    baybe_campaign = Quantity(
        type=JSON,
        description="""
        Contains the full JSON serialized BayBE campaign that represents this
        Bayesian Optimization.
        """,
    )

    def from_baybe(campaign):
        """Instantiate a BayesianOptimization from a BayBE campaign."""
        dictionary = campaign.to_dict()
        result = BayesianOptimization()
        result.baybe_campaign = dictionary.copy()

        searchspace: dict = dictionary.pop('searchspace', {})
        discrete: dict = searchspace.get('discrete', {})
        continuous: dict = searchspace.get('continuous', {})
        ps = discrete.get('parameters', []) + continuous.get('parameters', [])
        parameters = []
        for parameter in ps:
            if parameter['type'] == 'CategoricalParameter':
                parameters.append(
                    {
                        'm_def': (
                            'nomad_bayesian_optimization.schema_packages.'
                            'bayesian_optimization.CategoricalParameter'
                        ),
                        'name': parameter['name'],
                        'values': parameter['values'],
                    }
                )
            elif parameter['type'] == 'NumericalDiscreteParameter':
                parameters.append(
                    {
                        'm_def': (
                            'nomad_bayesian_optimization.schema_packages.'
                            'bayesian_optimization.NumericalDiscreteParameter'
                        ),
                        'name': parameter['name'],
                        'values': parameter['values'],
                    }
                )
            elif parameter['type'] == 'NumericalContinuousParameter':
                parameters.append(
                    {
                        'm_def': (
                            'nomad_bayesian_optimization.schema_packages.'
                            'bayesian_optimization.ContinuousParameter'
                        ),
                        'name': parameter['name'],
                        'lower_bound': parameter['bounds']['lower'],
                        'upper_bound': parameter['bounds']['upper'],
                    }
                )
            elif parameter['type'] == 'SubstanceParameter':
                raise NotImplementedError('SubstanceParameter not implemented')
        dictionary['parameters'] = parameters
        result.m_update_from_dict(dictionary)

        # Populate optimization steps
        df = deserialize_dataframe(dictionary['_measurements_exp'])
        optimization = Optimization(status='Finished')
        for i, step in df.iterrows():
            optimization.steps.append(Step(value_used=step.to_dict()))

        # Populate suggested step
        df = deserialize_dataframe(dictionary['_cached_recommendation'])
        if not df.empty:
            optimization.steps.append(Step(value_suggestion=df.to_dict()))

        result.optimization = optimization

        return result

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.optimization or not self.optimization.steps:
            return

        # Gather information from the steps into a single list
        figures = []
        steps = []
        for step in self.optimization.steps:
            value = step.used or step.value_recommended
            values = [value.get(parameter.name) for parameter in parameters]

        # Create a separate plot for each objective. TODO: The number of plots
        # to create should probably be limited, or at least the number that are
        # shown should be limited.
        targets = [self.objective.target]
        for target in targets:
            # Generate a plot that shows how the optimization progresses each step
            target_name = target.name
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=df['BatchNr'],
                    y=df[target_name],
                    mode='lines+markers',
                )
            )
            figure.update_layout(
                template='plotly_white',
                title='Progress',
                xaxis_title='Iteration',
                yaxis_title=target_name,
            )
            figures.append(
                PlotlyFigure(label='Progress', figure=figure.to_plotly_json())
            )

        # Create a table of the traversed search space from last to first step.
        # Recommended, but not yet tried values are added to the table as well.
        df = df[::-1]
        figure = go.Figure(
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
        figure.update_layout(
            template='plotly_white',
            margin=dict(l=0, r=0, t=0, b=0),
            width=800,
        )
        figures.append(figure)

        self.figures = figures


m_package.__init_metainfo__()
