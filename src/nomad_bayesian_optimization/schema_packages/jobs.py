# from nomad.datamodel.data import Schema
# from nomad.datamodel.metainfo.action import ActionSection
# from nomad.datamodel.metainfo.annotations import ELNAnnotation
# from nomad.metainfo import MEnum, MSection, Quantity, SchemaPackage, Section, SubSection

# m_package = SchemaPackage()


# class Job(MSection):
#     """Represents a single job."""

#     status = Quantity(
#         type=MEnum('Initialized', 'Finished', 'Error'),
#         description='Job status.',
#         a_eln=ELNAnnotation(
#             component='EnumEditQuantity',
#         ),
#     )
#     batch_id = Quantity(
#         type=str,
#         description='Batch identifier for this job',
#         a_eln=dict(component='StringEditQuantity'),
#     )


# class MyAction(ActionSection):
#     """Used to create jobs in a batch."""

#     action_trigger = Quantity(
#         description='Press to create a batch of jobs.',
#         a_eln=dict(component='ActionEditQuantity', label='Submit batch'),
#     )

#     def perform_action(self, archive, logger):
#         if self.action_trigger:
#             for i in range(self.n_jobs or 0):
#                 self.m_add_sub_section(
#                     Jobs.jobs, Job(batch_id=self.batch_id, status='Initialized')
#                 )


# class Jobs(MyAction, Schema):
#     """Used to create jobs in batches."""

#     m_def = Section(
#         a_eln=ELNAnnotation(
#             lane_width='600px',
#         )
#     )

#     batch_id = Quantity(
#         type=str,
#         description='Identifier for this batch',
#         a_eln=dict(component='StringEditQuantity'),
#     )
#     n_jobs = Quantity(
#         type=int,
#         description='How many jobs to create.',
#         a_eln=dict(component='NumberEditQuantity'),
#     )
#     jobs = SubSection(section_def=Job, repeats=True)

#     def normalize(self, archive, logger):
#         super().normalize(archive, logger)


# m_package.__init_metainfo__()
