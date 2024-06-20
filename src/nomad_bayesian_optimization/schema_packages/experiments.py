from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.metainfo import MEnum, Quantity, SchemaPackage

m_package = SchemaPackage()


class CVDExperiment(Schema):
    """Experiment for growing thin films using chemical vapor deposition."""

    substrate = Quantity(
        type=MEnum('Silicon carbide', 'Silicon', 'Gallium nitride'),
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.EnumEditQuantity,
        ),
    )
    gas_flow_rate = Quantity(
        type=float,
        unit='liter/minute',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
        description='Gas flow rate.',
    )
    temperature = Quantity(
        type=float,
        unit='K',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
        description='Temperature during experiment.',
    )
    refractive_index = Quantity(
        type=float,
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
        description='Refractive index of the thin-film sample.',
    )


m_package.__init_metainfo__()
