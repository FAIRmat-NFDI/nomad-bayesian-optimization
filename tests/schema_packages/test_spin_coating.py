from nomad.client import normalize_all
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.units import ureg

from nomad_bayesian_optimization.actions.campaign.models import (
    TargetSpec,
    VariableSpec,
)
from nomad_bayesian_optimization.measurement_reader import extract_record
from nomad_bayesian_optimization.schema_packages.spin_coating import SpinCoating


def _build_archive():
    archive = EntryArchive(metadata=EntryMetadata())
    archive.data = SpinCoating(
        film_thickness=120 * ureg.nanometer,
        precursor_concentration=1.2 * ureg.molar,
        spin_coating_speed=4000 * ureg.rpm,
        spinning_time=0.5 * ureg.minute,
    )
    return archive


def test_schema():
    """Values are stored in the lab units defined by the schema."""
    archive = _build_archive()
    normalize_all(archive)
    data = archive.data

    assert data.film_thickness.to('nanometer').magnitude == 120.0
    assert data.precursor_concentration.to('molar').magnitude == 1.2
    assert data.spin_coating_speed.to('rpm').magnitude == 4000.0
    assert data.spinning_time.to('second').magnitude == 30.0


def test_extract_measurement_record():
    """Bayesian optimization reads the magnitudes in the schema's lab units."""
    archive = _build_archive()
    variables = [
        VariableSpec(name='precursor_concentration', kind='continuous'),
        VariableSpec(name='spin_coating_speed', kind='continuous'),
        VariableSpec(name='spinning_time', kind='continuous'),
    ]
    targets = [TargetSpec(name='film_thickness', mode='MATCH', match_value=100.0)]

    record = extract_record(archive.data, variables, targets)

    assert record == {
        'precursor_concentration': 1.2,
        'spin_coating_speed': 4000.0,
        'spinning_time': 30.0,
        'film_thickness': 120.0,
    }
