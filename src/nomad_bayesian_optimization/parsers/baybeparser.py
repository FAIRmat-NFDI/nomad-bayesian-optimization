import json
from typing import Dict

from baybe.serialization.utils import deserialize_dataframe
from nomad.datamodel import EntryArchive
from nomad.parsing import MatchingParser

from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
    Step,
)


class BayBEParser(MatchingParser):
    """Parser for BayBE serialized Bayesian optimization campaigns."""

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives: Dict[str, EntryArchive] = None,
    ) -> None:
        with open(mainfile) as f:
            campaign = json.loads(f)

        """Instantiate a BayesianOptimization from a BayBE campaign."""
        dictionary = campaign.to_dict()
        result = BayesianOptimization()

        searchspace: dict = dictionary.pop('searchspace', {})
        discrete: dict = searchspace.get('discrete', {})
        continuous: dict = searchspace.get('continuous', {})
        ps = discrete.get('parameters', []) + continuous.get('parameters', [])
        parameters = []
        for parameter in ps:
            if parameter['type'] == 'CategoricalParameter':
                parameters.append(
                    {
                        'm_def': (
                            'nomad_bayesian_optimization.schema_packages.'
                            'bayesian_optimization.CategoricalParameter'
                        ),
                        'name': parameter['name'],
                        'values': parameter['values'],
                    }
                )
            elif parameter['type'] == 'NumericalDiscreteParameter':
                parameters.append(
                    {
                        'm_def': (
                            'nomad_bayesian_optimization.schema_packages.'
                            'bayesian_optimization.NumericalDiscreteParameter'
                        ),
                        'name': parameter['name'],
                        'values': parameter['values'],
                    }
                )
            elif parameter['type'] == 'NumericalContinuousParameter':
                parameters.append(
                    {
                        'm_def': (
                            'nomad_bayesian_optimization.schema_packages.'
                            'bayesian_optimization.ContinuousParameter'
                        ),
                        'name': parameter['name'],
                        'lower_bound': parameter['bounds']['lower'],
                        'upper_bound': parameter['bounds']['upper'],
                    }
                )
            elif parameter['type'] == 'SubstanceParameter':
                raise NotImplementedError('SubstanceParameter not implemented')
        dictionary['parameters'] = parameters
        result.m_update_from_dict(dictionary)

        # Populate optimization steps
        df = deserialize_dataframe(dictionary['_measurements_exp'])
        for i, step in df.iterrows():
            result.steps.append(Step(values_used=step.to_dict()))

        # Populate suggested step
        df = deserialize_dataframe(dictionary['_cached_recommendation'])
        if not df.empty:
            result.steps.append(Step(value_suggestion=df.to_dict()))

        archive.data = result
