"""Unit tests for the generic measurement reader.

These exercise the pure extraction helpers with lightweight stand-ins for
metainfo sections (a ``m_def`` with ``name``/``qualified_name`` and pint-like
quantity values), so they run without NOMAD or BayBE.
"""

from nomad_bayesian_optimization.actions.campaign.models import (
    TargetSpec,
    VariableSpec,
)
from nomad_bayesian_optimization.measurement_reader import (
    _matches_schema,
    _resolve_value,
    check_authorized,
    extract_record,
)


class FakeMDef:
    def __init__(self, name, qualified):
        self.name = name
        self._qualified = qualified

    def qualified_name(self):
        return self._qualified


class FakeQuantity:
    """Stand-in for a pint quantity carrying a unit."""

    def __init__(self, magnitude):
        self.magnitude = magnitude


class FakeSection:
    def __init__(self, m_def=None, **quantities):
        if m_def is not None:
            self.m_def = m_def
        for key, value in quantities.items():
            setattr(self, key, value)


class FakeUpload:
    def __init__(self, upload_id, main_author, coauthors=None):
        self.upload_id = upload_id
        self.main_author = main_author
        self.coauthors = coauthors or []


def test_resolve_value_handles_units_plain_and_nested():
    section = FakeSection(
        temperature=FakeQuantity(450.0),  # unit-carrying quantity
        refractive_index=2.05,  # plain scalar
        substrate='SiC',  # categorical
        sub=FakeSection(value=FakeQuantity(3.0)),  # nested path
    )
    assert _resolve_value(section, 'temperature') == 450.0
    assert _resolve_value(section, 'refractive_index') == 2.05
    assert _resolve_value(section, 'substrate') == 'SiC'
    assert _resolve_value(section, 'sub.value') == 3.0
    assert _resolve_value(section, 'missing') is None


def test_matches_schema_by_name_and_qualified_suffix():
    section = FakeSection(
        m_def=FakeMDef('MySample', 'my_plugin.schema_packages.mine.MySample')
    )
    assert _matches_schema(section, 'MySample')
    assert _matches_schema(section, 'schema_packages.mine.MySample')
    assert not _matches_schema(section, 'Other')
    assert not _matches_schema(None, 'MySample')


def test_extract_record_reads_variables_and_targets():
    section = FakeSection(
        m_def=FakeMDef('MySample', 'x.MySample'),
        temperature=FakeQuantity(450.0),
        substrate='SiC',
        refractive_index=2.05,
    )
    variables = [
        VariableSpec(name='temperature', kind='continuous'),
        VariableSpec(name='substrate', kind='categorical'),
    ]
    targets = [TargetSpec(name='refractive_index', mode='MATCH')]
    assert extract_record(section, variables, targets) == {
        'temperature': 450.0,
        'substrate': 'SiC',
        'refractive_index': 2.05,
    }


def test_extract_record_uses_quantity_path_override():
    section = FakeSection(
        params=FakeSection(temp=FakeQuantity(300.0)),
        result=FakeQuantity(1.5),
    )
    variables = [
        VariableSpec(name='temperature', quantity='params.temp', kind='continuous')
    ]
    targets = [TargetSpec(name='y', quantity='result', mode='MAX')]
    assert extract_record(section, variables, targets) == {
        'temperature': 300.0,
        'y': 1.5,
    }


def test_extract_record_skips_incomplete_rows():
    section = FakeSection(temperature=FakeQuantity(450.0))  # no target present
    variables = [VariableSpec(name='temperature', kind='continuous')]
    targets = [TargetSpec(name='refractive_index', mode='MATCH')]
    # Missing target -> dropped when required.
    assert extract_record(section, variables, targets) is None
    # ...but kept when targets are not required (e.g. reading a suggestion).
    assert extract_record(section, variables, targets, require_targets=False) == {
        'temperature': 450.0
    }
    # Missing a required variable -> dropped regardless.
    empty = FakeSection(refractive_index=2.0)
    assert extract_record(empty, variables, targets, require_targets=False) is None


def test_check_authorized():
    upload = FakeUpload('up1', main_author='alice', coauthors=['bob'])
    check_authorized(upload, 'alice')  # main author: ok
    check_authorized(upload, 'bob')  # coauthor: ok
    try:
        check_authorized(upload, 'eve')
    except PermissionError as exc:
        assert 'eve' in str(exc)
    else:  # pragma: no cover
        raise AssertionError('expected PermissionError')
