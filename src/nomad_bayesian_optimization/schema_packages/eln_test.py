from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.metainfo import MEnum, Quantity, SchemaPackage

m_package = SchemaPackage()


class ELNTest(Schema):
    """Schema for testing different ELN features."""

    enum = Quantity(
        type=MEnum('Silicon carbide', 'Silicon', 'Gallium nitride'),
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.EnumEditQuantity,
        ),
        description='Enum',
    )
    number = Quantity(
        type=float,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
        description='Number',
    )
    number_with_limits = Quantity(
        type=float,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity, minValue=0, maxValue=10
        ),
        description='Number with limit.',
    )
    number_with_unit_and_limit = Quantity(
        type=float,
        unit='liter/minute',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity, minValue=0, maxValue=10
        ),
        description='Number with unit and limit.',
    )
    number_with_unit_and_limit_and_displayunit = Quantity(
        type=float,
        unit='liter/minute',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            minValue=0,
            maxValue=10,
            defaultDisplayUnit='liter/hour',
        ),
        description='Number with unit, limit and display unit.',
    )


m_package.__init_metainfo__()
m_package.__init_metainfo__()
m_package.__init_metainfo__()
