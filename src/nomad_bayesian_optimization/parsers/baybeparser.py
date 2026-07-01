import json

from nomad.datamodel import EntryArchive
from nomad.parsing import MatchingParser

from nomad_bayesian_optimization.campaign_converter import (
    campaign_dict_to_schema_dict,
)
from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
)


class BayBEParser(MatchingParser):
    """Parser for BayBE serialized Bayesian optimization campaigns."""

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives: dict[str, EntryArchive] = None,
    ) -> None:
        with open(mainfile) as f:
            campaign = json.load(f)

        # Decoding the serialized measurement dataframes requires BayBE (and
        # pandas) to be importable. Import lazily so that the parser entry point
        # can still be loaded in environments without the (heavy) BayBE stack.
        try:
            schema_dict = campaign_dict_to_schema_dict(campaign)
        except ImportError as exc:
            if logger is not None:
                logger.error(
                    'Could not parse BayBE campaign because BayBE is not '
                    'installed. Install the "parsing" extra of this plugin.',
                    exc_info=exc,
                )
            return

        archive.data = BayesianOptimization.m_from_dict(schema_dict)
