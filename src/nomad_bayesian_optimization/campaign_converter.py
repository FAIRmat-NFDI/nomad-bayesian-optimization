"""Shared conversion from a serialized BayBE campaign into a dictionary that
conforms to the :class:`BayesianOptimization` NOMAD schema.

The function :func:`campaign_dict_to_schema_dict` is used both by the BayBE
parser (:mod:`nomad_bayesian_optimization.parsers.baybeparser`) and by the
Bayesian optimization action
(:mod:`nomad_bayesian_optimization.actions.campaign.activities`)
so that both produce identical archives.

The input is the plain dictionary obtained from ``json.loads(campaign.to_json())``.
BayBE serializes every polymorphic node (parameter, objective, target,
transformation, recommender) with a ``"type"`` key holding its class name and
uses field aliases (``targets``, ``values``, ``target``, ``bounds``, ``data``,
``encoding``), which makes the dictionary straightforward to walk. The only
values that require BayBE itself are the dataframes (measurements,
recommendations and the discrete search space candidates), which are stored as
base64-encoded pickles and are decoded lazily with
``baybe.serialization.utils.deserialize_dataframe``. The short names of
acquisition functions are also looked up lazily from BayBE when available.
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA = 'nomad_bayesian_optimization.schema_packages.bayesian_optimization'

# BayBE class names grouped by how they map onto the schema parameter classes.
# Note: parameters in the *discrete* subspace carry a ``"type"`` field, whereas
# parameters in the *continuous* subspace do not (there is only one continuous
# parameter class), so continuous parameters are handled by location instead.
_CATEGORICAL_TYPES = {'CategoricalParameter', 'TaskParameter'}
_NUMERICAL_DISCRETE_TYPES = {'NumericalDiscreteParameter'}
_SUBSTANCE_TYPES = {'SubstanceParameter'}


def _m(name: str) -> str:
    """Return the fully qualified ``m_def`` for a schema class."""
    return f'{SCHEMA}.{name}'


def _dataframe_records(serialized_df: Any) -> list[dict]:
    """Decode a serialized BayBE dataframe into a list of JSON-safe records.

    ``serialized_df`` is the value found under ``_measurements_exp`` /
    ``_cached_recommendation`` (a base64-encoded pickle string, or the
    ``{"constructor": ...}`` dict form). Returns an empty list for missing or
    empty dataframes. The round-trip through ``DataFrame.to_json`` converts
    numpy scalars / NaNs into plain JSON types so the result can be stored in a
    NOMAD ``JSON`` quantity.

    Importing BayBE may raise ``ImportError``; callers handle that (the parser
    logs and bails so the entry point still loads without the BayBE stack).
    """
    if serialized_df is None:
        return []

    from baybe.serialization.utils import deserialize_dataframe

    df = deserialize_dataframe(serialized_df)
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient='records'))


def measured_records(campaign: dict) -> list[dict]:
    """Return one record per recorded measurement of the campaign."""
    return _dataframe_records(campaign.get('measurements_exp'))


def recommended_records(campaign: dict) -> list[dict]:
    """Return one record per pending (recommended but not measured) suggestion."""
    return _dataframe_records(campaign.get('cached_recommendation'))


def _extract_bounds(transformation: dict | None) -> dict | None:
    """Best-effort extraction of an indicative interval from a transformation.

    BayBE transformations no longer carry explicit target bounds, so we derive
    an indicative range where possible:

    - Bell transformations expose a ``center`` and ``sigma`` (range center ± sigma).
    - Ramp/triangular transformations expose ``cutoffs`` (min/max of the cutoffs).

    Returns ``None`` when no meaningful interval can be derived.
    """
    if not isinstance(transformation, dict):
        return None

    center = transformation.get('center')
    sigma = transformation.get('sigma')
    if center is not None and sigma is not None:
        return {'m_def': _m('Bounds'), 'lower': center - sigma, 'upper': center + sigma}

    cutoffs = transformation.get('cutoffs')
    if isinstance(cutoffs, dict):
        lower = cutoffs.get('lower')
        upper = cutoffs.get('upper')
        if lower is not None and upper is not None:
            return {'m_def': _m('Bounds'), 'lower': lower, 'upper': upper}
    if isinstance(cutoffs, (list, tuple)) and len(cutoffs) == 2:  # noqa: PLR2004
        return {'m_def': _m('Bounds'), 'lower': cutoffs[0], 'upper': cutoffs[1]}

    return None


def _interval_center(interval: Any) -> float | None:
    """Return the center of an interval given as a ``(lower, upper)`` pair or dict."""
    if isinstance(interval, dict):
        interval = (interval.get('lower'), interval.get('upper'))
    if isinstance(interval, (list, tuple)) and len(interval) == 2:  # noqa: PLR2004
        lower, upper = interval
        if lower is not None and upper is not None:
            return (lower + upper) / 2
    return None


def _target_goal(target: dict) -> dict:
    """Determine the goal (``mode``) of a serialized BayBE target.

    Match targets are transformed so that BayBE can maximize or minimize the
    transformed value, which means that ``minimize`` alone does not tell what
    happens to the measured values. The goal is therefore read from the
    ``constructor_info`` that BayBE stores for targets created with e.g.
    ``NumericalTarget.match_bell``. Targets created without such a constructor
    (e.g. with the legacy ``MATCH`` mode) are recognized from their bell or
    triangular transformation. Returns an empty dict when the goal cannot be
    determined (e.g. for sigmoid or custom chained transformations).
    """
    minimize = bool(target.get('minimize'))
    info = target.get('constructor_info')
    info = info if isinstance(info, dict) else {}
    transformation = target.get('transformation')
    transformation = transformation if isinstance(transformation, dict) else {}
    transformation_type = transformation.get('type')

    if str(info.get('constructor', '')).startswith('match_'):
        match_value = info.get('match_value')
        if match_value is None:
            # ``match_triangular`` may be given only cutoffs, centered on the value.
            match_value = _interval_center(info.get('cutoffs'))
        return {
            'mode': 'MISMATCH' if info.get('mismatch_instead') else 'MATCH',
            'match_value': match_value,
            'match_mode': info.get('match_mode') or '=',
        }

    match_value_keys = {
        'BellTransformation': 'center',
        'TriangularTransformation': 'peak',
    }
    if transformation_type in match_value_keys:
        return {
            'mode': 'MISMATCH' if minimize else 'MATCH',
            'match_value': transformation.get(match_value_keys[transformation_type]),
            'match_mode': '=',
        }

    if transformation_type in (None, 'IdentityTransformation'):
        return {'mode': 'MIN' if minimize else 'MAX'}
    return {}


def _convert_discrete_parameter(parameter: dict) -> dict | None:
    """Convert a single serialized BayBE *discrete* parameter into a schema dict."""
    ptype = parameter.get('type')
    name = parameter.get('name')

    if ptype in _CATEGORICAL_TYPES:
        return {
            'm_def': _m('CategoricalParameter'),
            'name': name,
            'values': [str(v) for v in parameter.get('values', [])],
            'encoding': parameter.get('encoding'),
        }
    if ptype in _NUMERICAL_DISCRETE_TYPES:
        return {
            'm_def': _m('NumericalDiscreteParameter'),
            'name': name,
            'values': parameter.get('values', []),
            'tolerance': parameter.get('tolerance'),
        }
    if ptype in _SUBSTANCE_TYPES:
        data = parameter.get('data', {}) or {}
        return {
            'm_def': _m('SubstanceParameter'),
            'name': name,
            'values': [
                {'m_def': _m('BoSubstance'), 'name': key, 'smiles': smiles}
                for key, smiles in data.items()
            ],
            'encoding': parameter.get('encoding'),
        }

    # Unknown/unsupported parameter type: keep at least the name so nothing is
    # silently dropped.
    if name is not None:
        return {'m_def': _m('Parameter'), 'name': name}
    return None


def _convert_continuous_parameter(parameter: dict) -> dict:
    """Convert a single serialized BayBE *continuous* parameter into a schema dict."""
    bounds = parameter.get('bounds', {}) or {}
    return {
        'm_def': _m('ContinuousParameter'),
        'name': parameter.get('name'),
        'lower_bound': bounds.get('lower'),
        'upper_bound': bounds.get('upper'),
    }


def _convert_target(target: dict, weight: float | None = None) -> dict:
    """Convert a single serialized BayBE target into a schema dict."""
    transformation = target.get('transformation')
    transformation_type = None
    if isinstance(transformation, dict):
        transformation_type = transformation.get('type')

    result = {
        'm_def': _m('Target'),
        'type': target.get('type'),
        'name': target.get('name'),
        'minimize': target.get('minimize'),
        'transformation': transformation_type,
        'transformation_parameters': transformation,
        'constructor_info': target.get('constructor_info'),
        'weight': weight,
        **_target_goal(target),
    }
    bounds = _extract_bounds(transformation)
    if bounds is not None:
        result['bounds'] = bounds
    return result


def _convert_objective(objective: dict | None) -> dict | None:
    """Convert a serialized BayBE objective into a schema dict."""
    if not objective:
        return None

    otype = objective.get('type')
    result: dict = {'m_def': _m('Objective'), 'type': otype}

    if otype == 'SingleTargetObjective':
        target = objective.get('target')
        if target:
            result['targets'] = [_convert_target(target)]
    else:
        # DesirabilityObjective / ParetoObjective (and any future multi-target
        # objective) expose a list of targets, optionally with weights.
        targets = objective.get('targets', []) or []
        weights = objective.get('weights') or [None] * len(targets)
        result['scalarizer'] = objective.get('scalarizer')
        result['targets'] = [
            _convert_target(target, weight) for target, weight in zip(targets, weights)
        ]

    return result


def _convert_surrogate(surrogate: dict | None) -> dict | None:
    """Convert a serialized BayBE surrogate model into a schema dict.

    BayBE wraps the surrogate in a ``CompositeSurrogate`` that replicates a single
    template surrogate for each target; in that case the template is reported.
    """
    if not isinstance(surrogate, dict):
        return None
    surrogates = surrogate.get('surrogates')
    if (
        surrogate.get('type') == 'CompositeSurrogate'
        and isinstance(surrogates, dict)
        and surrogates.get('type') == '_ReplicationMapping'
        and isinstance(surrogates.get('template'), dict)
    ):
        surrogate = surrogates['template']

    result = {'m_def': _m('SurrogateModel'), 'type': surrogate.get('type')}
    kernel = surrogate.get('kernel_or_factory')
    if isinstance(kernel, dict):
        result['kernel_factory'] = {
            'm_def': _m('KernelFactory'),
            'type': kernel.get('type'),
        }
    return result


def _acquisition_abbreviation(type_name: str) -> str | None:
    """Return BayBE's short name (e.g. ``qLogEI``) for an acquisition function.

    Returns ``None`` when BayBE is not installed or the class is unknown.
    """
    try:
        import baybe.acquisition
    except ImportError:
        return None
    return getattr(getattr(baybe.acquisition, type_name, None), 'abbreviation', None)


def _convert_acquisition_function(
    acquisition_function: dict | None, objective_type: str | None
) -> dict:
    """Convert a serialized BayBE acquisition function into a schema dict.

    When no acquisition function is set, BayBE picks a default depending on the
    objective (see ``BayesianRecommender._get_acquisition_function``), which is
    reported instead.
    """
    if isinstance(acquisition_function, dict):
        type_name = acquisition_function.get('type')
    elif objective_type == 'ParetoObjective':
        type_name = 'qLogNoisyExpectedHypervolumeImprovement'
    else:
        type_name = 'qLogExpectedImprovement'
    return {
        'm_def': _m('AcquisitionFunction'),
        'type': type_name,
        'abbreviation': _acquisition_abbreviation(type_name),
    }


def _convert_recommender(
    recommender: dict | None,
    objective_type: str | None = None,
    include_config: bool = True,
) -> dict | None:
    """Convert a serialized BayBE recommender into a schema dict.

    The main building blocks (nested recommenders of a meta recommender, surrogate
    model, acquisition function) are extracted, while the full serialized
    configuration is stored as JSON on the top-level recommender, as recommenders
    come in many variants.
    """
    if not recommender or not isinstance(recommender, dict):
        return None
    result: dict = {'m_def': _m('Recommender'), 'type': recommender.get('type')}
    if include_config:
        result['config'] = recommender

    for key in ('initial_recommender', 'recommender'):
        nested = _convert_recommender(
            recommender.get(key), objective_type, include_config=False
        )
        if nested is not None:
            result[key] = nested

    surrogate = _convert_surrogate(recommender.get('surrogate_model'))
    if surrogate is not None:
        result['surrogate_model'] = surrogate

    # Only Bayesian recommenders have an acquisition function (possibly ``None``,
    # meaning that the BayBE default is used).
    if 'acquisition_function' in recommender:
        result['acquisition_function'] = _convert_acquisition_function(
            recommender['acquisition_function'], objective_type
        )

    for key in ('switch_after', 'hybrid_sampler', 'sampling_percentage'):
        if recommender.get(key) is not None:
            result[key] = recommender[key]
    return result


def _search_space_type(discrete: dict, continuous: dict) -> str | None:
    """Return the search space type, following BayBE's ``SearchSpace.type``."""
    has_discrete = bool(discrete.get('parameters'))
    has_continuous = bool(continuous.get('parameters'))
    if has_discrete and has_continuous:
        return 'Hybrid'
    if has_discrete:
        return 'Discrete'
    if has_continuous:
        return 'Continuous'
    return None


def discrete_candidate_count(campaign: dict) -> int | None:
    """Return the number of candidates in the discrete subspace of the campaign.

    Returns ``None`` when the campaign has no discrete subspace. Like
    :func:`measured_records`, this requires BayBE to decode the serialized
    dataframe.
    """
    discrete = (campaign.get('searchspace', {}) or {}).get('discrete', {}) or {}
    if not discrete.get('parameters') or discrete.get('exp_rep') is None:
        return None

    from baybe.serialization.utils import deserialize_dataframe

    df = deserialize_dataframe(discrete['exp_rep'])
    return None if df is None else len(df)


def campaign_dict_to_schema_dict(campaign: dict, status: str | None = None) -> dict:
    """Convert a serialized BayBE campaign dict into a ``BayesianOptimization`` dict.

    Args:
        campaign: The dictionary from ``json.loads(campaign.to_json())``.
        status: Optional status to record (e.g. ``'Finished'``). When omitted,
            the schema default is used.

    Returns:
        A dictionary (with ``m_def`` keys) that can be assigned to
        ``archive.data`` or used with ``BayesianOptimization.m_from_dict``.
    """
    searchspace = campaign.get('searchspace', {}) or {}
    discrete = searchspace.get('discrete', {}) or {}
    continuous = searchspace.get('continuous', {}) or {}

    parameters = []
    for param in discrete.get('parameters', []):
        converted = _convert_discrete_parameter(param)
        if converted is not None:
            parameters.append(converted)
    for param in continuous.get('parameters', []):
        parameters.append(_convert_continuous_parameter(param))

    # Steps are not built here: each step is stored with a generated, typed schema
    # (see :mod:`nomad_bayesian_optimization.step_schema`), which the parser
    # attaches to ``archive.definitions`` and instantiates from the decoded
    # measurement/recommendation records (:func:`measured_records`,
    # :func:`recommended_records`).
    objective = campaign.get('objective')
    result: dict = {
        'm_def': _m('BayesianOptimization'),
        'parameters': parameters,
        'objective': _convert_objective(objective),
        'recommender': _convert_recommender(
            campaign.get('recommender'),
            objective_type=(objective or {}).get('type'),
        ),
    }
    search_space_type = _search_space_type(discrete, continuous)
    if search_space_type is not None:
        result['search_space_type'] = search_space_type
    if campaign.get('n_batches_done') is not None:
        result['n_batches_done'] = campaign['n_batches_done']
    if status is not None:
        result['status'] = status
    return result
