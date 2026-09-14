"""On-the-fly generation of a typed per-campaign step schema.

Every Bayesian optimization campaign has a different search space, so the values
recorded at each step cannot be modelled by a single static schema. Instead we
generate, at parse time, a subclass of :class:`Step` (``CampaignStep``) whose
quantities mirror the campaign's variables and targets, attach it to
``EntryArchive.definitions``, and store each step as a typed instance of it.

The quantity ``type`` is inferred from the BayBE parameters/targets so that a
plain serialized campaign (without action metadata) still yields a typed schema.
When the Bayesian optimization action injects field metadata (resolved from the
actual measurement-schema quantities), the corresponding ``type``, ``unit`` and
``description`` are used instead, so the generated schema faithfully mirrors the
source fields.

The three public helpers are used by the parser and unit-tested directly:

- :func:`derive_step_fields` — campaign dict (+ optional metadata) -> field specs.
- :func:`attach_step_package` — build the ``CampaignStep`` section and attach it
  to ``archive.definitions``.
- :func:`make_step_instance` — build one typed step instance from a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from nomad.metainfo import Package, Quantity, Section

from nomad_bayesian_optimization.naming import sanitize_quantity_name
from nomad_bayesian_optimization.schema_packages.bayesian_optimization import Step

# BayBE parameter classes whose measured values are labels (strings) rather than
# numbers. Everything else (numerical discrete, continuous, targets) is numeric.
_STR_PARAM_TYPES = {'CategoricalParameter', 'TaskParameter', 'SubstanceParameter'}

# Coarse type name -> metainfo quantity type used when generating the schema.
_METAINFO_TYPES: dict[str, Any] = {
    'float': np.float64,
    'int': np.int64,
    'bool': bool,
    'str': str,
}

# Coarse type name -> caster applied to a raw record value before storing it, so
# that e.g. a float coming from the BayBE dataframe is stored in an int quantity.
_CASTERS = {'float': float, 'int': int, 'bool': bool, 'str': str}


@dataclass
class FieldSpec:
    """One generated step quantity, derived from a campaign variable/target."""

    name: str  # original BayBE name / dataframe column
    quantity_name: str  # sanitized metainfo quantity name
    type: str  # coarse type: 'float' | 'int' | 'str' | 'bool'
    unit: str | None = None
    description: str | None = None
    is_target: bool = False


def _iter_target_dicts(objective: dict | None):
    """Yield each serialized target dict of a BayBE objective."""
    if not objective:
        return
    if objective.get('type') == 'SingleTargetObjective':
        target = objective.get('target')
        if target:
            yield target
        return
    for target in objective.get('targets', []) or []:
        if target:
            yield target


def derive_step_fields(
    campaign: dict, field_meta: dict | None = None
) -> list[FieldSpec]:
    """Derive the ordered list of step quantities for a campaign.

    Args:
        campaign: The serialized BayBE campaign dict.
        field_meta: Optional ``{baybe_name: {type, unit, description}}`` mapping
            injected by the action (resolved from the measurement schema). When a
            name is present, its type/unit/description override the inferred type.

    Returns:
        One :class:`FieldSpec` per declared variable and target (in that order).
    """
    field_meta = field_meta or {}
    searchspace = campaign.get('searchspace', {}) or {}
    discrete = (searchspace.get('discrete', {}) or {}).get('parameters', []) or []
    continuous = (searchspace.get('continuous', {}) or {}).get('parameters', []) or []

    specs: list[FieldSpec] = []
    seen: set[str] = set()

    def add(name: str | None, base_type: str, is_target: bool) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        meta = field_meta.get(name) or {}
        specs.append(
            FieldSpec(
                name=name,
                quantity_name=sanitize_quantity_name(name),
                type=meta.get('type') or base_type,
                unit=meta.get('unit'),
                description=meta.get('description'),
                is_target=is_target,
            )
        )

    for param in discrete:
        base = 'str' if param.get('type') in _STR_PARAM_TYPES else 'float'
        add(param.get('name'), base, False)
    for param in continuous:
        add(param.get('name'), 'float', False)
    for target in _iter_target_dicts(campaign.get('objective')):
        add(target.get('name'), 'float', True)

    return specs


def attach_step_package(archive, field_specs: list[FieldSpec]) -> Section:
    """Create the ``CampaignStep`` section and attach it to ``archive.definitions``.

    The section subclasses the existing :class:`Step`; one quantity is added per
    field spec, carrying the field's type, unit and description. The original
    BayBE column name is stored in the quantity's ``more`` dict under
    ``baybe_name`` so ``normalize`` can map the quantities back to dataframe
    columns.

    Returns the generated :class:`Section` definition (use ``.section_cls`` to
    instantiate it).
    """
    section_def = Section(name='CampaignStep', base_sections=[Step.m_def])
    for spec in field_specs:
        quantity = Quantity(
            name=spec.quantity_name,
            type=_METAINFO_TYPES.get(spec.type, str),
            unit=spec.unit or None,
            description=spec.description or None,
        )
        quantity.more['baybe_name'] = spec.name
        section_def.quantities.append(quantity)

    package = Package(name='BayesianOptimizationCampaign')
    package.section_definitions.append(section_def)

    # Mark the package as an in-archive (custom) package when the archive is bound
    # to an upload/entry, so it is not registered in the global package registry.
    metadata = getattr(archive, 'metadata', None)
    if metadata is not None:
        package.upload_id = getattr(metadata, 'upload_id', None)
        package.entry_id = getattr(metadata, 'entry_id', None)

    package.init_metainfo()
    archive.definitions = package
    return section_def


def make_step_instance(
    section_def: Section,
    record: dict,
    field_specs: list[FieldSpec],
    *,
    recommended: bool,
):
    """Instantiate one ``CampaignStep`` from a measurement/recommendation record."""
    instance = section_def.section_cls()
    instance.recommended = recommended
    for spec in field_specs:
        value = record.get(spec.name)
        if value is None:
            continue
        caster = _CASTERS.get(spec.type)
        if caster is not None:
            try:
                value = caster(value)
            except (TypeError, ValueError):
                pass
        setattr(instance, spec.quantity_name, value)
    return instance
