"""Tests for resolving step-field metadata from a measurement schema.

The action resolves each variable/target ``quantity`` dot-path to a metainfo
``Quantity`` definition and reads its type/unit/description. These tests exercise
the resolution helpers against a small in-memory schema, without an upload.
"""

import logging

import numpy as np
from nomad.metainfo import MSection, Quantity, SchemaPackage, SubSection

from nomad_bayesian_optimization.measurement_reader import (
    _coarse_type,
    _data_section_path,
    _matches_schema,
    _resolve_quantity_def,
    iter_matching_data_sections,
)

m_package = SchemaPackage()


class SubData(MSection):
    refractive_index = Quantity(type=np.float64, description='Refractive index')


class MySample(MSection):
    temperature = Quantity(
        type=np.float64, unit='kelvin', description='Sample temperature'
    )
    label = Quantity(type=str)
    count = Quantity(type=int)
    sub = SubSection(section_def=SubData)


m_package.__init_metainfo__()


def test_resolve_quantity_def_and_coarse_type():
    section_def = MySample.m_def

    temperature = _resolve_quantity_def(section_def, 'temperature')
    assert temperature is not None
    assert _coarse_type(temperature) == 'float'
    assert str(temperature.unit) == 'kelvin'
    assert temperature.description == 'Sample temperature'

    assert _coarse_type(_resolve_quantity_def(section_def, 'label')) == 'str'
    assert _coarse_type(_resolve_quantity_def(section_def, 'count')) == 'int'


def test_resolve_archive_root_relative_paths():
    """Paths given relative to the archive root resolve on the data section."""
    section_def = MySample.m_def

    temperature = _resolve_quantity_def(
        section_def, _data_section_path('data.temperature')
    )
    assert temperature is not None
    assert str(temperature.unit) == 'kelvin'

    nested = _resolve_quantity_def(
        section_def, _data_section_path('data.sub.refractive_index')
    )
    assert nested is not None
    assert _coarse_type(nested) == 'float'


def test_resolve_nested_and_missing_paths():
    section_def = MySample.m_def

    nested = _resolve_quantity_def(section_def, 'sub.refractive_index')
    assert nested is not None
    assert _coarse_type(nested) == 'float'

    assert _resolve_quantity_def(section_def, 'does_not_exist') is None
    assert _resolve_quantity_def(section_def, 'sub.nope') is None


def test_matches_schema_is_lenient():
    sample = MySample()
    # Matches on the class name even when the full qualified name differs.
    assert _matches_schema(sample, 'MySample')
    # Matches the exact qualified name too.
    assert _matches_schema(sample, sample.m_def.qualified_name())
    # A genuinely different schema does not match.
    assert not _matches_schema(sample, 'SomethingElse')
    assert not _matches_schema(None, 'MySample')


class _FakeEntry:
    def __init__(self, entry_id):
        self.entry_id = entry_id


class _FakeArchive:
    def __init__(self, data):
        self.data = data


class _FakeUpload:
    upload_id = 'upload1'

    def __init__(self, entries):
        self.successful_entries = entries

    def get_entry(self, entry_id):
        return _FakeEntry(entry_id)


class _FakeContext:
    def __init__(self, archives):
        self._archives = archives  # entry_id -> archive

    def load_archive(self, entry_id, upload_id, installation_url):
        return self._archives[entry_id]


def test_iter_matching_data_sections_finds_matches():
    sample = MySample()
    upload = _FakeUpload([_FakeEntry('e1'), _FakeEntry('e2')])
    context = _FakeContext({'e1': _FakeArchive(sample), 'e2': _FakeArchive(None)})

    matched = list(
        iter_matching_data_sections(
            upload, context, 'MySample', logger=logging.getLogger('t')
        )
    )
    assert matched == [sample]


def test_iter_matching_data_sections_warns_when_nothing_matches(caplog):
    sample = MySample()
    upload = _FakeUpload([_FakeEntry('e1')])
    context = _FakeContext({'e1': _FakeArchive(sample)})

    with caplog.at_level(logging.WARNING):
        matched = list(
            iter_matching_data_sections(upload, context, 'NoSuchSchema')
        )
    assert matched == []
    # The warning names the requested schema and the schemas actually seen.
    assert any(
        'NoSuchSchema' in rec.message and 'MySample' in rec.message
        for rec in caplog.records
    )
