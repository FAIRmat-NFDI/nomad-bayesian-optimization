"""Read measurement records from NOMAD entries into BayBE-ready dictionaries.

Given an upload, a target schema name and the variable/target specifications of a
campaign, :func:`read_measurement_records` enumerates the matching entries and
extracts one record per entry. Each record is keyed by the BayBE variable/target
``name`` so it can be fed straight into ``pandas.DataFrame`` and
``Campaign.add_measurements``.

This replaces the CVD-specific extraction of the legacy action and works for any
schema by resolving each variable/target ``quantity`` path against the entry's
``data`` section.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nomad_bayesian_optimization.actions.campaign.models import (
        TargetSpec,
        VariableSpec,
    )

_default_logger = logging.getLogger(__name__)


def _get_logger(logger: Any) -> Any:
    """Return the given logger, or a module-level default when none is passed."""
    return logger if logger is not None else _default_logger


def check_authorized(upload: Any, user_id: str) -> None:
    """Raise ``PermissionError`` unless the user may access the upload."""
    is_coauthor = isinstance(upload.coauthors, list) and user_id in upload.coauthors
    if not (upload.main_author == user_id or is_coauthor):
        raise PermissionError(
            f'User {user_id} is not authorized to access upload {upload.upload_id}.'
        )


def _resolve_value(section: Any, path: str) -> Any:
    """Resolve a dot-separated quantity ``path`` against a metainfo section."""
    obj = section
    for part in path.split('.'):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    if obj is None:
        return None
    # Quantities carrying a unit come back as pint quantities; use the magnitude.
    magnitude = getattr(obj, 'magnitude', None)
    if magnitude is not None:
        return float(magnitude)
    return obj


def _data_section_path(path: str) -> str:
    """Turn an archive-root-relative quantity path into a data-section-relative one.

    Quantity paths are given the way quantities are addressed elsewhere in NOMAD,
    i.e. relative to the archive root (``data.molecular_mass``), but they are
    resolved against the entry's ``data`` section. A path without the ``data.``
    prefix is used as it is.
    """
    prefix = 'data.'
    return path[len(prefix) :] if path.startswith(prefix) else path


def _schema_candidates(data_section: Any) -> set[str]:
    """Return the names a ``data`` section can be matched against."""
    if data_section is None:
        return set()
    m_def = data_section.m_def
    candidates: set[str] = {getattr(m_def, 'name', None)}
    try:
        candidates.add(m_def.qualified_name())
    except Exception:  # noqa: BLE001 - qualified_name is best-effort
        pass
    return {c for c in candidates if c}


def _matches_schema(data_section: Any, schema_name: str) -> bool:
    """Return whether a ``data`` section is an instance of the target schema.

    Matches on exact name, a dotted-suffix of the qualified name, or the final
    dotted segment (class name), so a short ``schema_name`` like ``TADFMolecule``
    matches ``…tadf_molecules.TADFMolecule`` regardless of the module path given.
    """
    schema_leaf = schema_name.split('.')[-1]
    return any(
        candidate == schema_name
        or candidate.endswith(schema_name)
        or candidate.split('.')[-1] == schema_leaf
        for candidate in _schema_candidates(data_section)
    )


def extract_record(
    data_section: Any,
    variables: list[VariableSpec],
    targets: list[TargetSpec],
    *,
    require_targets: bool = True,
) -> dict[str, Any] | None:
    """Extract a single measurement record from a ``data`` section.

    Returns ``None`` when a required variable is missing, or when
    ``require_targets`` is set and no target value could be read.
    """
    record: dict[str, Any] = {}
    for variable in variables:
        path = _data_section_path(variable.quantity or variable.name)
        value = _resolve_value(data_section, path)
        if value is None:
            return None
        record[variable.name] = value

    has_target = False
    for target in targets:
        path = _data_section_path(target.quantity or target.name)
        value = _resolve_value(data_section, path)
        if value is not None:
            record[target.name] = value
            has_target = True

    if require_targets and not has_target:
        return None
    return record


def _resolve_quantity_def(section_def: Any, path: str) -> Any:
    """Resolve a dot-separated quantity ``path`` to its metainfo Quantity definition.

    Walks ``all_sub_sections`` for the intermediate parts and looks the final part
    up in ``all_quantities``. Returns ``None`` when the path cannot be resolved.
    """
    current = section_def
    parts = path.split('.')
    for part in parts[:-1]:
        sub = current.all_sub_sections.get(part) if current is not None else None
        if sub is None:
            return None
        target = sub.sub_section
        current = target.m_resolved() if hasattr(target, 'm_resolved') else target
    if current is None:
        return None
    return current.all_quantities.get(parts[-1])


def _coarse_type(quantity_def: Any) -> str:
    """Map a metainfo Quantity's data type onto a coarse 'float'|'int'|'str'|'bool'."""
    dtype = getattr(quantity_def, 'type', None)
    std = None
    if dtype is not None and hasattr(dtype, 'standard_type'):
        try:
            std = dtype.standard_type()
        except Exception:  # noqa: BLE001 - standard_type is best-effort
            std = None
    if not std:
        return 'str'
    if std.startswith(('float', 'complex')):
        return 'float'
    if std.startswith(('int', 'uint')):
        return 'int'
    if std == 'bool':
        return 'bool'
    return 'str'


def iter_matching_data_sections(
    upload: Any,
    context: Any,
    schema_name: str,
    *,
    entry_ids: list[str] | None = None,
    logger: Any = None,
):
    """Yield the ``data`` section of every entry that matches ``schema_name``.

    Shared by :func:`read_measurement_records` and :func:`resolve_field_metadata`.
    Unlike the previous silent implementation, load failures are logged (not
    swallowed without trace) and a single warning is emitted when nothing matches,
    naming the requested ``schema_name`` and the distinct schema names actually
    seen — so a mismatch (or unprocessed entries) is diagnosable from the worker
    log instead of surfacing as mysteriously empty results.
    """
    logger = _get_logger(logger)

    if entry_ids is not None:
        entries = [upload.get_entry(entry_id) for entry_id in entry_ids]
    else:
        entries = list(upload.successful_entries)

    logger.info(
        f'Scanning {len(entries)} entr'
        f'{"y" if len(entries) == 1 else "ies"} for schema {schema_name!r}.'
    )

    matched = 0
    seen: set[str] = set()
    for entry in entries:
        if entry is None:
            continue
        try:
            archive = context.load_archive(entry.entry_id, upload.upload_id, None)
        except Exception as exc:  # noqa: BLE001 - skip but record unreadable archives
            logger.warning(
                f'Could not load archive for entry {entry.entry_id}: {exc!r}'
            )
            continue
        data_section = getattr(archive, 'data', None)
        if data_section is None:
            continue
        if _matches_schema(data_section, schema_name):
            matched += 1
            yield data_section
        else:
            seen.update(_schema_candidates(data_section))

    if matched == 0:
        logger.warning(
            f'No entry matched schema {schema_name!r}; schemas seen: '
            f'{sorted(seen) or "none"}. Field metadata and seeding will be empty.'
        )


def resolve_field_metadata(
    upload: Any,
    user_id: str,
    schema_name: str,
    variables: list[VariableSpec],
    targets: list[TargetSpec],
    logger: Any = None,
) -> dict[str, dict[str, Any]]:
    """Resolve type/unit/description for each variable/target from the schema.

    Loads one entry matching ``schema_name`` and reads the metainfo ``Quantity``
    definition each variable/target ``quantity`` path points to. The result is a
    ``{baybe_name: {'type', 'unit', 'description'}}`` mapping the action injects
    into the persisted campaign so the parser can generate a faithfully typed step
    schema. Returns an empty mapping when no matching entry/field is found (the
    parser then falls back to inferring types from the BayBE parameters).
    """
    check_authorized(upload, user_id)
    logger = _get_logger(logger)

    from nomad.datamodel.context import ServerContext

    context = ServerContext(upload)

    section_def = None
    for data_section in iter_matching_data_sections(
        upload, context, schema_name, logger=logger
    ):
        section_def = data_section.m_def
        break

    if section_def is None:
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for spec in [*variables, *targets]:
        path = spec.quantity or spec.name
        quantity_def = _resolve_quantity_def(section_def, _data_section_path(path))
        if quantity_def is None:
            unresolved.append(path)
            continue
        unit = getattr(quantity_def, 'unit', None)
        metadata[spec.name] = {
            'type': _coarse_type(quantity_def),
            'unit': str(unit) if unit is not None else None,
            'description': getattr(quantity_def, 'description', None) or None,
        }
    if unresolved:
        logger.warning(
            f'Could not resolve quantity path(s) {unresolved} on schema '
            f'{schema_name!r}; those fields get no unit/description.'
        )
    return metadata


def read_measurement_records(
    upload: Any,
    user_id: str,
    schema_name: str,
    variables: list[VariableSpec],
    targets: list[TargetSpec],
    entry_ids: list[str] | None = None,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Read measurement records from an upload's entries.

    Args:
        upload: The ``nomad.processing.data.Upload`` to read from.
        user_id: The requesting user (checked for authorization).
        schema_name: Name (or fully qualified suffix) of the measurement schema.
        variables: The campaign variables (search-space dimensions).
        targets: The campaign targets.
        entry_ids: Restrict reading to these entries. When ``None`` all successful
            entries of the upload are considered (used to seed a fresh campaign).
        logger: Optional logger (e.g. the Temporal ``activity.logger``).

    Returns:
        A list of records keyed by variable/target ``name``.
    """
    check_authorized(upload, user_id)
    logger = _get_logger(logger)

    from nomad.datamodel.context import ServerContext

    context = ServerContext(upload)

    records: list[dict[str, Any]] = []
    for data_section in iter_matching_data_sections(
        upload, context, schema_name, entry_ids=entry_ids, logger=logger
    ):
        record = extract_record(data_section, variables, targets)
        if record is not None:
            records.append(record)

    if not records:
        logger.warning(
            f'No measurement records read for schema {schema_name!r} '
            f'(entries matched but no complete record could be extracted, or '
            f'no entry matched).'
        )
    return records
