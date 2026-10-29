from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity, SchemaPackage

m_package = SchemaPackage()


class SpinCoating(Schema):
    """A spin coating measurement: the thickness of the film obtained for a given
    precursor concentration, spin coating speed and spinning time.
    """

    film_thickness = Quantity(
        type=float,
        unit='nanometer',
        description='Thickness of the deposited film.',
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )
    precursor_concentration = Quantity(
        type=float,
        unit='molar',
        description='Concentration of the precursor in the coating solution.',
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )
    spin_coating_speed = Quantity(
        type=float,
        unit='rpm',
        description='Rotational speed of the substrate during spinning.',
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )
    spinning_time = Quantity(
        type=float,
        unit='second',
        description='Duration of the spinning step.',
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )


m_package.__init_metainfo__()
