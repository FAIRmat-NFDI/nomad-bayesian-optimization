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
values that require BayBE itself are the measurement/recommendation dataframes,
which are stored as base64-encoded pickles and are decoded lazily with
``baybe.serialization.utils.deserialize_dataframe``.
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
    """
    if serialized_df is None:
        return []

    from baybe.serialization.utils import deserialize_dataframe

    df = deserialize_dataframe(serialized_df)
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient='records'))


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
        'weight': weight,
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


def _convert_recommender(recommender: dict | None) -> dict | None:
    """Convert a serialized BayBE recommender into a schema dict.

    Recommenders can be deeply nested and come in many variants, so we store the
    class name plus the full serialized configuration as JSON rather than
    modelling every field.
    """
    if not recommender:
        return None
    return {
        'm_def': _m('Recommender'),
        'type': recommender.get('type'),
        'config': recommender,
    }


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

    # Build one step per recorded measurement, plus a final step for a pending
    # (recommended but not yet measured) recommendation. BayBE serializes these
    # dataframes under keys without the leading underscore of the attribute name.
    steps = [
        {'m_def': _m('Step'), 'values_used': record}
        for record in _dataframe_records(campaign.get('measurements_exp'))
    ]
    for record in _dataframe_records(campaign.get('cached_recommendation')):
        steps.append({'m_def': _m('Step'), 'values_recommended': record})

    result: dict = {
        'm_def': _m('BayesianOptimization'),
        'parameters': parameters,
        'objective': _convert_objective(campaign.get('objective')),
        'recommender': _convert_recommender(campaign.get('recommender')),
        'steps': steps,
        'n_steps': len(steps),
    }
    if status is not None:
        result['status'] = status
    return result
