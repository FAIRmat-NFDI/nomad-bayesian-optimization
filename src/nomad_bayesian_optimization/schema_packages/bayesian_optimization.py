import pandas as pd
import plotly.graph_objects as go
from nomad.datamodel.data import ArchiveSection, Schema
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.metainfo import (
    JSON,
    MEnum,
    MSection,
    Quantity,
    SchemaPackage,
    SubSection,
)

m_package = SchemaPackage()


class Parameter(ArchiveSection):
    """A single optimization parameter (one dimension of the search space)."""

    name = Quantity(
        type=str,
    )
    value_reference = Quantity(
        type=Quantity,
    )
    definition = Quantity(
        type=str,
    )


class ContinuousParameter(Parameter):
    """A continuous numerical parameter defined by a closed interval."""

    lower_bound = Quantity(
        type=float,
    )
    upper_bound = Quantity(
        type=float,
    )


class DiscreteParameter(Parameter):
    """Base class for parameters with a finite set of allowed values."""

    pass


class NumericalDiscreteParameter(DiscreteParameter):
    """A numerical parameter with a finite set of allowed values."""

    values = Quantity(
        type=float,
        shape=['*'],
    )
    tolerance = Quantity(
        type=float,
        description="""
        Maximum allowed deviation of a measured value from a discrete value for
        it to still be recognized as that value.
        """,
    )


class CategoricalParameter(DiscreteParameter):
    """A parameter with a finite set of categorical (labelled) values."""

    values = Quantity(
        type=str,
        shape=['*'],
    )
    encoding = Quantity(
        type=str,
        description='Encoding used to represent the categories numerically.',
    )


class BoSubstance(ArchiveSection):
    """A chemical substance identified by a name and a SMILES string."""

    name = Quantity(
        type=str,
    )
    smiles = Quantity(
        type=str,
    )


class SubstanceParameter(DiscreteParameter):
    """A parameter whose values are chemical substances (name + SMILES)."""

    values = SubSection(
        section_def=BoSubstance,
        repeats=True,
    )
    encoding = Quantity(
        type=str,
        description="""
        Encoding used to represent the substances numerically (e.g. MORDRED,
        RDKIT, MORGAN_FP).
        """,
    )


class Bounds(MSection):
    """A numerical interval."""

    lower = Quantity(
        type=float,
    )
    upper = Quantity(type=float)


class Target(MSection):
    """A single optimization target."""

    type = Quantity(
        type=str,
        description='The BayBE target class name (e.g. NumericalTarget).',
    )
    name = Quantity(type=str)
    minimize = Quantity(
        type=bool,
        description='Whether the target is minimized (True) or maximized (False).',
    )
    transformation = Quantity(
        type=str,
        description="""
        Name of the transformation applied to the raw target values before
        optimization (e.g. BellTransformation, AffineTransformation).
        """,
    )
    transformation_parameters = Quantity(
        type=JSON,
        description='Full serialized transformation, including its parameters.',
    )
    weight = Quantity(
        type=float,
        description="""
        Relative weight of this target within a multi-target (desirability)
        objective.
        """,
    )
    bounds = SubSection(section_def=Bounds)


class Objective(MSection):
    """The optimization objective, possibly combining several targets."""

    type = Quantity(
        type=MEnum(
            'SingleTargetObjective',
            'DesirabilityObjective',
            'ParetoObjective',
        )
    )
    scalarizer = Quantity(
        type=str,
        description="""
        Scalarizer used to combine multiple targets in a desirability objective
        (e.g. GEOM_MEAN, MEAN).
        """,
    )
    targets = SubSection(section_def=Target, repeats=True)


class KernelFactory(MSection):
    type = Quantity(type=str)


class SurrogateModel(MSection):
    type = Quantity(type=str)
    kernel_factory = SubSection(section_def=KernelFactory)


class AcquisitionFunction(MSection):
    type = Quantity(type=str)


class Recommender(MSection):
    """The recommender (strategy) used to suggest new experiments."""

    type = Quantity(type=str)
    surrogate_model = SubSection(section_def=SurrogateModel)
    initial_recommender = SubSection(section_def='Recommender')
    recommender = SubSection(section_def='Recommender')
    acquisition_function = SubSection(section_def=AcquisitionFunction)
    hybrid_sampler = Quantity(type=str)
    sampling_percentage = Quantity(type=float)
    config = Quantity(
        type=JSON,
        description='Full serialized recommender configuration.',
    )


class Step(MSection):
    """Represents a single step in the Bayesian optimization procedure. Is used to
    record the actually used parameter values and observed objective values during
    optimization (not necessarily the ones that the optimization procedure originally
    recommended).

    Since every optimization run differs in its design, the concrete per-parameter and
    per-target values are stored on a generated subclass of this section
    (``CampaignStep``) whose quantities mirror the campaign's variables and targets.
    That generated schema is created on the fly and stored under
    ``EntryArchive.definitions``, so the values can be recorded with proper types,
    units and descriptions. This base class only holds fields common to every step.
    """

    recommended = Quantity(
        type=bool,
        description="""
        Whether this step is a pending recommendation suggested by the optimizer
        (True) that has not yet been measured, as opposed to a recorded measurement
        (False).
        """,
    )


class BayesianOptimization(PlotSection, Schema):
    """Represents a single Bayesian optimization task."""

    status = Quantity(
        type=MEnum('Initializing', 'Suggesting', 'Acquiring', 'Finished', 'Error'),
        default='Initializing',
        description='Optimization status.',
    )
    parameters = SubSection(section_def=Parameter, repeats=True)
    objective = SubSection(section_def=Objective)
    recommender = SubSection(section_def=Recommender)
    steps = SubSection(section_def=Step, repeats=True)
    n_steps = Quantity(
        type=int,
        description='Number of steps in optimization.',
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        self.n_steps = len(self.steps or [])

        if not self.steps:
            return

        # Gather the values from each step into a single dataframe. The step values
        # live on a generated ``CampaignStep`` subclass whose quantities each carry
        # the original BayBE column name in their ``more`` dict.
        steps_list = []
        for step in self.steps:
            row = {}
            for name, quantity in step.m_def.all_quantities.items():
                column = quantity.more.get('baybe_name') if quantity.more else None
                if not column:
                    continue
                value = getattr(step, name, None)
                if value is None:
                    continue
                # Quantities carrying a unit come back as pint quantities.
                magnitude = getattr(value, 'magnitude', None)
                if magnitude is not None:
                    value = float(magnitude)
                row[column] = value
            if row:
                steps_list.append(row)
        if not steps_list:
            return
        steps_df = pd.DataFrame(steps_list)

        # Determine the x-axis for the progress plot: use the BayBE batch number
        # if it is available, otherwise fall back to a running step index.
        if 'BatchNr' in steps_df.columns:
            x_values = steps_df['BatchNr']
        else:
            x_values = list(range(1, len(steps_df) + 1))

        # Create a separate progress plot for each target of the objective.
        figures = []
        targets = (self.objective.targets if self.objective else None) or []
        for target in targets:
            target_name = target.name
            if not target_name or target_name not in steps_df.columns:
                continue
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=steps_df[target_name],
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
                PlotlyFigure(label=target_name, figure=figure.to_plotly_json())
            )

        # Create a table of the traversed search space from last to first step.
        # Recommended, but not yet tried values are added to the table as well.
        table_df = steps_df[::-1]
        figure = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(table_df.columns),
                        align='left',
                    ),
                    cells=dict(
                        values=table_df.transpose().values.tolist(),
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
        figures.append(
            PlotlyFigure(label='Search space', figure=figure.to_plotly_json())
        )
        self.figures = figures


m_package.__init_metainfo__()
