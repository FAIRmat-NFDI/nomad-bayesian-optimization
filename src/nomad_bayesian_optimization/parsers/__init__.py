from nomad.config.models.plugins import ParserEntryPoint


class BayBEParserEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.parsers.baybeparser import BayBEParser

        return BayBEParser(**self.dict())


baybe = BayBEParserEntryPoint(
    name='BayBE Parser',
    description='Used to parse serialized BayBE campaigns into NOMAD entries.',
    mainfile_name_re='.*\.json',
    mainfile_contents_dict={'__has_key': 'searchspace'},
)
