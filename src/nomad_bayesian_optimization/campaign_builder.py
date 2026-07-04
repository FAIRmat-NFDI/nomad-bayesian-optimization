"""Build a BayBE :class:`~baybe.Campaign` from a generic optimization spec.

This is the counterpart of
:func:`nomad_bayesian_optimization.campaign_converter.campaign_dict_to_schema_dict`:
the converter turns a serialized BayBE campaign into the ``BayesianOptimization``
schema, whereas this module turns a user-provided
:class:`~nomad_bayesian_optimization.actions.campaign.models.BayesianOptimizationInput`
(``actions.campaign.models``, describing variables and targets) into a fresh
BayBE campaign.

BayBE is imported lazily inside the functions so that this module can be imported
in environments without the (heavy) BayBE stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from baybe import Campaign

    from nomad_bayesian_optimization.actions.campaign.models import (
        BayesianOptimizationInput,
        TargetSpec,
        VariableSpec,
    )


def _build_parameter(variable: VariableSpec) -> Any:
    """Convert a single :class:`VariableSpec` into a BayBE parameter."""
    from baybe.parameters import (
        CategoricalParameter,
        NumericalContinuousParameter,
        NumericalDiscreteParameter,
        SubstanceParameter,
    )

    kind = variable.kind
    if kind == 'continuous':
        if variable.lower_bound is None or variable.upper_bound is None:
            raise ValueError(
                f"Continuous variable '{variable.name}' requires both "
                'lower_bound and upper_bound.'
            )
        return NumericalContinuousParameter(
            name=variable.name,
            bounds=(variable.lower_bound, variable.upper_bound),
        )
    if kind == 'numerical_discrete':
        if not variable.values:
            raise ValueError(
                f"Numerical discrete variable '{variable.name}' requires values."
            )
        return NumericalDiscreteParameter(
            name=variable.name,
            values=[float(v) for v in variable.values],
            tolerance=variable.tolerance or 0.0,
        )
    if kind == 'categorical':
        if not variable.values:
            raise ValueError(
                f"Categorical variable '{variable.name}' requires values."
            )
        return CategoricalParameter(
            name=variable.name,
            values=list(variable.values),
            encoding=variable.encoding or 'OHE',
        )
    if kind == 'substance':
        data = _parse_substances(variable)
        parameter_kwargs: dict[str, Any] = {'name': variable.name, 'data': data}
        if variable.encoding:
            parameter_kwargs['encoding'] = variable.encoding
        return SubstanceParameter(**parameter_kwargs)

    raise ValueError(f"Unknown variable kind '{kind}' for variable '{variable.name}'.")


def _parse_substances(variable: VariableSpec) -> dict[str, str]:
    """Parse ``name=SMILES`` substance values into a ``{name: SMILES}`` mapping."""
    if not variable.values:
        raise ValueError(
            f"Substance variable '{variable.name}' requires values of the form "
            "'name=SMILES'."
        )
    data: dict[str, str] = {}
    for value in variable.values:
        name, sep, smiles = value.partition('=')
        if not sep or not name.strip() or not smiles.strip():
            raise ValueError(
                f"Substance value '{value}' for variable '{variable.name}' must be "
                "of the form 'name=SMILES'."
            )
        data[name.strip()] = smiles.strip()
    return data


def _build_target(target: TargetSpec) -> Any:
    """Convert a single :class:`TargetSpec` into a BayBE target."""
    from baybe.targets import NumericalTarget

    mode = target.mode
    if mode == 'MAX':
        return NumericalTarget(name=target.name)
    if mode == 'MIN':
        return NumericalTarget(name=target.name, minimize=True)
    if mode == 'MATCH':
        if target.match_value is None or target.sigma is None:
            raise ValueError(
                f"Target '{target.name}' in MATCH mode requires match_value and "
                'sigma.'
            )
        return NumericalTarget.match_bell(
            name=target.name,
            match_value=target.match_value,
            sigma=target.sigma,
        )
    raise ValueError(f"Unknown target mode '{mode}' for target '{target.name}'.")


def build_campaign(spec: BayesianOptimizationInput) -> Campaign:
    """Build a fresh BayBE campaign from the action input.

    Args:
        spec: The action input describing the variables and targets.

    Returns:
        A BayBE :class:`~baybe.Campaign` with no measurements yet.
    """
    from baybe import Campaign
    from baybe.objectives import (
        DesirabilityObjective,
        ParetoObjective,
        SingleTargetObjective,
    )
    from baybe.searchspace import SearchSpace

    if not spec.variables:
        raise ValueError('At least one variable is required.')
    if not spec.targets:
        raise ValueError('At least one target is required.')

    parameters = [_build_parameter(variable) for variable in spec.variables]
    searchspace = SearchSpace.from_product(parameters)

    targets = [_build_target(target) for target in spec.targets]
    objective = _build_objective(
        targets,
        DesirabilityObjective=DesirabilityObjective,
        ParetoObjective=ParetoObjective,
        SingleTargetObjective=SingleTargetObjective,
        weights=[target.weight for target in spec.targets],
        scalarizer=spec.scalarizer,
    )

    # Rely on BayBE's default recommender (TwoPhaseMetaRecommender with a
    # BotorchRecommender), which handles discrete, continuous and hybrid search
    # spaces as well as single-, desirability- and Pareto-objectives.
    return Campaign(searchspace, objective)


def _build_objective(
    targets: list[Any],
    *,
    DesirabilityObjective: Any,
    ParetoObjective: Any,
    SingleTargetObjective: Any,
    weights: list[float | None],
    scalarizer: str | None,
) -> Any:
    """Choose and build the BayBE objective from the targets.

    - a single target uses a ``SingleTargetObjective``;
    - multiple targets with explicit weights or a scalarizer use a
      ``DesirabilityObjective`` (which requires the targets to be normalized to a
      non-negative range, e.g. via MATCH mode);
    - multiple plain (maximize/minimize) targets use a ``ParetoObjective``.
    """
    if len(targets) == 1:
        return SingleTargetObjective(target=targets[0])

    use_desirability = scalarizer is not None or any(
        weight is not None for weight in weights
    )
    if use_desirability:
        return DesirabilityObjective(
            targets=targets,
            weights=[weight if weight is not None else 1.0 for weight in weights],
            scalarizer=scalarizer or 'GEOM_MEAN',
        )
    return ParetoObjective(targets=targets)
