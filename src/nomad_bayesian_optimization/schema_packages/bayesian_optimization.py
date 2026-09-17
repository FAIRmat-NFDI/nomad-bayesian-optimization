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
    mode = Quantity(
        type=MEnum('MAX', 'MIN', 'MATCH', 'MISMATCH'),
        description="""
        Goal for the measured (untransformed) target values: maximize, minimize,
        match the match value or stay away from it. Not set when the goal cannot be
        determined from the target transformation.
        """,
    )
    match_value = Quantity(
        type=float,
        description='The value to be matched (MATCH) or avoided (MISMATCH).',
    )
    match_mode = Quantity(
        type=MEnum('=', '>=', '<='),
        description="""
        Matching mode: with '>=' ('<=') all values above (below) the match value are
        considered a match.
        """,
    )
    minimize = Quantity(
        type=bool,
        description="""
        Whether the transformed target value is minimized (True) or maximized
        (False). For transformed targets this differs from the goal for the measured
        values, see ``mode``.
        """,
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
    constructor_info = Quantity(
        type=JSON,
        description="""
        The BayBE constructor used to create the target (e.g. match_bell) and its
        arguments.
        """,
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
    abbreviation = Quantity(
        type=str,
        description='Short name of the acquisition function (e.g. qLogEI).',
    )


class Recommender(MSection):
    """The recommender (strategy) used to suggest new experiments."""

    type = Quantity(type=str)
    surrogate_model = SubSection(section_def=SurrogateModel)
    initial_recommender = SubSection(section_def='Recommender')
    recommender = SubSection(section_def='Recommender')
    switch_after = Quantity(
        type=int,
        description="""
        Number of batches after which a meta recommender switches from the initial
        recommender to the main recommender.
        """,
    )
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
    search_space_type = Quantity(
        type=MEnum('Discrete', 'Continuous', 'Hybrid'),
        description="""
        Type of the search space: purely discrete, purely continuous or a hybrid of
        both.
        """,
    )
    n_candidates = Quantity(
        type=int,
        description="""
        Number of candidates in the discrete part of the search space. Not set for
        purely continuous search spaces.
        """,
    )
    parameters = SubSection(section_def=Parameter, repeats=True)
    objective = SubSection(section_def=Objective)
    recommender = SubSection(section_def=Recommender)
    steps = SubSection(section_def=Step, repeats=True)
    n_steps = Quantity(
        type=int,
        description='Number of steps in optimization.',
    )
    n_measurements = Quantity(
        type=int,
        description='Number of steps with recorded measurements.',
    )
    n_pending_recommendations = Quantity(
        type=int,
        description='Number of recommended steps that have not yet been measured.',
    )
    n_batches_done = Quantity(
        type=int,
        description='Number of measurement batches added to the campaign.',
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        steps = self.steps or []
        self.n_steps = len(steps)
        self.n_pending_recommendations = sum(bool(step.recommended) for step in steps)
        self.n_measurements = self.n_steps - self.n_pending_recommendations

        # Create a separate progress plot for each target of the objective. The
        # individual steps are not plotted: they are shown as a table directly from
        # the ``steps`` subsection.
        targets = (self.objective.targets if self.objective else None) or []
        self.figures = [
            figure for target in targets if (figure := self._progress_figure(target))
        ]

    def _find_step_quantity(self, name: str):
        """Return the step quantity that stores the given BayBE variable/target.

        The step values live on a generated ``CampaignStep`` subclass whose
        quantities each carry the original BayBE column name in their ``more`` dict.
        """
        if not self.steps:
            return None
        for quantity in self.steps[0].m_def.all_quantities.values():
            if quantity.more and quantity.more.get('baybe_name') == name:
                return quantity
        return None

    def _progress_figure(self, target: Target) -> PlotlyFigure | None:
        """Create a plot of the measured target values against the step number."""
        quantity = self._find_step_quantity(target.name)
        if quantity is None:
            return None

        # Recommended steps have not been measured yet and are left out. The step
        # number is the position in ``steps``, matching the order of the steps table.
        steps, values = [], []
        for number, step in enumerate(self.steps, start=1):
            value = getattr(step, quantity.name, None)
            if step.recommended or value is None:
                continue
            # Quantities carrying a unit come back as pint quantities.
            steps.append(number)
            values.append(float(getattr(value, 'magnitude', value)))
        if not values:
            return None

        figure = go.Figure(
            data=[
                go.Scatter(x=steps, y=values, mode='markers', name='Recorded values'),
                go.Scatter(
                    x=steps,
                    y=_best_so_far(values, target),
                    mode='lines',
                    line_shape='hv',
                    line_dash='dash',
                    name='Best so far',
                ),
            ]
        )
        if target.mode in ('MATCH', 'MISMATCH') and target.match_value is not None:
            figure.add_trace(
                go.Scatter(
                    x=[steps[0], steps[-1]],
                    y=[target.match_value] * 2,
                    mode='lines',
                    line_dash='dot',
                    name='Match value',
                )
            )
        y_title = target.name
        if quantity.unit is not None:
            y_title = f'{target.name} ({quantity.unit:~P})'
        figure.update_layout(
            template='plotly_white',
            xaxis_title='Step',
            xaxis_tickformat='d',
            yaxis_title=y_title,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        )
        return PlotlyFigure(label=target.name, figure=figure.to_plotly_json())


def _best_so_far(values: list[float], target: Target) -> list[float]:
    """Return the running best of the given target values.

    For MATCH (MISMATCH) targets the best value is the one closest to (farthest
    from) the match value, where values beyond the match value count as exact
    matches for the '>=' and '<=' match modes. Otherwise the best value is the
    largest or smallest one. When the target mode is unknown, the ``minimize`` flag
    is used, so the result is a best-effort estimate for such targets.
    """
    mode = target.mode
    if mode is None:
        mode = 'MIN' if target.minimize else 'MAX'

    def score(value: float) -> float:
        """Return a score for the value, higher is better."""
        if mode in ('MATCH', 'MISMATCH') and target.match_value is not None:
            difference = value - target.match_value
            if target.match_mode == '>=':
                difference = min(difference, 0.0)
            elif target.match_mode == '<=':
                difference = max(difference, 0.0)
            return abs(difference) if mode == 'MISMATCH' else -abs(difference)
        return -value if mode == 'MIN' else value

    best_values = []
    for value in values:
        best = best_values[-1] if best_values else value
        best_values.append(value if score(value) > score(best) else best)
    return best_values


m_package.__init_metainfo__()
