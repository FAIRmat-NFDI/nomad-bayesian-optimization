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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nomad_bayesian_optimization.actions.campaign.models import (
        TargetSpec,
        VariableSpec,
    )


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


def _matches_schema(data_section: Any, schema_name: str) -> bool:
    """Return whether a ``data`` section is an instance of the target schema."""
    if data_section is None:
        return False
    m_def = data_section.m_def
    candidates = {getattr(m_def, 'name', None)}
    try:
        candidates.add(m_def.qualified_name())
    except Exception:  # noqa: BLE001 - qualified_name is best-effort
        pass
    return any(
        candidate and (candidate == schema_name or candidate.endswith(schema_name))
        for candidate in candidates
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
        value = _resolve_value(data_section, variable.quantity or variable.name)
        if value is None:
            return None
        record[variable.name] = value

    has_target = False
    for target in targets:
        value = _resolve_value(data_section, target.quantity or target.name)
        if value is not None:
            record[target.name] = value
            has_target = True

    if require_targets and not has_target:
        return None
    return record


def read_measurement_records(
    upload: Any,
    user_id: str,
    schema_name: str,
    variables: list[VariableSpec],
    targets: list[TargetSpec],
    entry_ids: list[str] | None = None,
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

    Returns:
        A list of records keyed by variable/target ``name``.
    """
    check_authorized(upload, user_id)

    from nomad.datamodel.context import ServerContext

    context = ServerContext(upload)

    if entry_ids is not None:
        entries = [upload.get_entry(entry_id) for entry_id in entry_ids]
    else:
        entries = list(upload.successful_entries)

    records: list[dict[str, Any]] = []
    for entry in entries:
        if entry is None:
            continue
        try:
            archive = context.load_archive(entry.entry_id, upload.upload_id, None)
        except Exception:  # noqa: BLE001 - skip unreadable/incompatible archives
            continue
        data_section = getattr(archive, 'data', None)
        if not _matches_schema(data_section, schema_name):
            continue
        record = extract_record(data_section, variables, targets)
        if record is not None:
            records.append(record)
    return records
