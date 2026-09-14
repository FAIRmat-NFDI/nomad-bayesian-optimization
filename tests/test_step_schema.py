"""Unit tests for the on-the-fly step schema generation.

These tests do not require BayBE: they operate on plain campaign dicts and the
metainfo objects the helpers build.
"""

from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.context import ClientContext

from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
)
from nomad_bayesian_optimization.step_schema import (
    attach_step_package,
    derive_step_fields,
    make_step_instance,
)

CAMPAIGN = {
    'searchspace': {
        'discrete': {
            'parameters': [
                {'type': 'NumericalDiscreteParameter', 'name': 'pressure'},
                {'type': 'CategoricalParameter', 'name': 'substrate type'},
            ]
        },
        'continuous': {'parameters': [{'name': 'temperature'}]},
    },
    'objective': {'type': 'SingleTargetObjective', 'target': {'name': 'yield'}},
}


def test_derive_step_fields_infers_types():
    """Without metadata, types are inferred from the BayBE parameters/targets."""
    specs = {s.name: s for s in derive_step_fields(CAMPAIGN)}

    assert set(specs) == {'pressure', 'substrate type', 'temperature', 'yield'}
    assert specs['pressure'].type == 'float'
    assert specs['temperature'].type == 'float'  # continuous -> float
    assert specs['substrate type'].type == 'str'  # categorical -> str
    assert specs['yield'].type == 'float'  # target -> float
    assert specs['yield'].is_target is True
    assert specs['pressure'].is_target is False

    # Free-form names are sanitized into valid identifiers.
    assert specs['substrate type'].quantity_name == 'substrate_type'

    # No metadata -> no unit/description.
    assert specs['pressure'].unit is None
    assert specs['pressure'].description is None


def test_derive_step_fields_enriches_from_metadata():
    """Injected metadata overrides the type and adds unit/description."""
    field_meta = {
        'temperature': {
            'type': 'float',
            'unit': 'kelvin',
            'description': 'Substrate temperature',
        },
        'pressure': {'type': 'int', 'unit': 'pascal', 'description': None},
    }
    specs = {s.name: s for s in derive_step_fields(CAMPAIGN, field_meta)}

    assert specs['temperature'].unit == 'kelvin'
    assert specs['temperature'].description == 'Substrate temperature'
    assert specs['pressure'].type == 'int'  # overridden from inferred float
    assert specs['pressure'].unit == 'pascal'


def test_attach_step_package_builds_subclass_of_step():
    """The generated section subclasses ``Step`` and carries typed quantities."""
    field_meta = {
        'pressure': {'type': 'float', 'unit': 'pascal', 'description': 'Pressure'},
    }
    specs = derive_step_fields(CAMPAIGN, field_meta)

    archive = EntryArchive(metadata=EntryMetadata())
    archive.data = BayesianOptimization()
    step_def = attach_step_package(archive, specs)

    assert archive.definitions is not None
    assert archive.definitions.section_definitions[0] is step_def
    assert step_def.name == 'CampaignStep'
    assert any(base.name == 'Step' for base in step_def.base_sections)

    quantities = {q.name: q for q in step_def.quantities}
    assert set(quantities) == {'pressure', 'substrate_type', 'temperature', 'yield'}
    assert quantities['pressure'].type.standard_type() == 'float64'
    assert str(quantities['pressure'].unit) == 'pascal'
    assert quantities['pressure'].description == 'Pressure'
    assert quantities['substrate_type'].type.standard_type() == 'str'
    # The original BayBE column name is stored for normalize to map back.
    assert quantities['substrate_type'].more['baybe_name'] == 'substrate type'

    # The instantiable class really derives from the base Step section.
    from nomad_bayesian_optimization.schema_packages.bayesian_optimization import Step

    assert issubclass(step_def.section_cls, Step)


def test_make_step_instance_sets_values_and_flag():
    """Values are stored on the instance; missing values and columns are skipped."""
    specs = derive_step_fields(CAMPAIGN)
    archive = EntryArchive(metadata=EntryMetadata())
    archive.data = BayesianOptimization()
    step_def = attach_step_package(archive, specs)

    record = {
        'pressure': 1.5,
        'substrate type': 'Si',
        'temperature': 400.0,
        'yield': 60.0,
        'BatchNr': 3,  # not a declared field -> ignored
    }
    step = make_step_instance(step_def, record, specs, recommended=False)
    assert step.pressure == 1.5
    assert getattr(step, 'substrate_type') == 'Si'
    assert getattr(step, 'yield') == 60.0
    assert bool(step.recommended) is False

    # A recommendation without a target value leaves that quantity unset.
    recommendation = {'pressure': 2.0, 'substrate type': 'GaN', 'temperature': 450.0}
    pending = make_step_instance(step_def, recommendation, specs, recommended=True)
    assert bool(pending.recommended) is True
    assert getattr(pending, 'yield') is None


def test_generated_schema_survives_archive_roundtrip():
    """Serializing and reloading resolves the generated schema and step values.

    When the archive carries an upload/entry id (as during real processing) the
    generated package is a custom in-archive package, so each step's ``m_def`` is
    serialized as a local ``#/definitions/...`` reference that resolves on reload.
    """
    field_meta = {'pressure': {'type': 'float', 'unit': 'pascal', 'description': 'P'}}
    specs = derive_step_fields(CAMPAIGN, field_meta)

    context = ClientContext()
    archive = EntryArchive(
        m_context=context,
        metadata=EntryMetadata(upload_id='u1', entry_id='e1'),
    )
    archive.data = BayesianOptimization()
    step_def = attach_step_package(archive, specs)
    assert archive.definitions.m_is_custom_package is True
    archive.data.steps.append(
        make_step_instance(
            step_def,
            {'pressure': 1.0, 'substrate type': 'Si', 'temperature': 400.0},
            specs,
            recommended=False,
        )
    )

    serialized = archive.m_to_dict()
    step_m_def = serialized['data']['steps'][0]['m_def']
    assert step_m_def == '#/definitions/section_definitions/0'

    reloaded = EntryArchive.m_from_dict(serialized, m_context=context)
    assert len(reloaded.data.steps) == 1
    assert reloaded.data.steps[0].pressure.magnitude == 1.0
    assert getattr(reloaded.data.steps[0], 'substrate_type') == 'Si'
    assert reloaded.definitions.section_definitions[0].name == 'CampaignStep'
