"""Helpers for turning BayBE variable/target names into valid metainfo names.

BayBE parameter/target names are free-form strings, but metainfo ``Quantity``
names must be valid Python identifiers (they become attribute names on the
generated ``CampaignStep`` section). :func:`sanitize_quantity_name` maps a
free-form name deterministically onto such an identifier so that the converter
(which builds the section definition and the step values) and the schema's
``normalize`` (which reads them back) agree on the same names.
"""

from __future__ import annotations

import re

_INVALID = re.compile(r'\W')


def sanitize_quantity_name(name: str) -> str:
    """Return a valid metainfo quantity name derived from ``name``.

    Non-identifier characters are replaced by underscores and a leading
    underscore is added when the result would otherwise start with a digit.
    The mapping is deterministic so encoders and decoders stay in sync.
    """
    sanitized = _INVALID.sub('_', str(name))
    if not sanitized:
        return '_'
    if sanitized[0].isdigit():
        sanitized = f'_{sanitized}'
    return sanitized
