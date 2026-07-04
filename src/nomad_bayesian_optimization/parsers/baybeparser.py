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

        # The Bayesian optimization action injects a top-level ``status`` key when
        # it persists a campaign; use it if present (a plain serialized campaign
        # has no status and falls back to the schema default).
        status = campaign.get('status')

        # Decoding the serialized measurement dataframes requires BayBE (and
        # pandas) to be importable. Import lazily so that the parser entry point
        # can still be loaded in environments without the (heavy) BayBE stack.
        try:
            schema_dict = campaign_dict_to_schema_dict(campaign, status=status)
        except ImportError as exc:
            if logger is not None:
                logger.error(
                    'Could not parse BayBE campaign because BayBE is not '
                    'installed. Install the "parsing" extra of this plugin.',
                    exc_info=exc,
                )
            return

        archive.data = BayesianOptimization.m_from_dict(schema_dict)
