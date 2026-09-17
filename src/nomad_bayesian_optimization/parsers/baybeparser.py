import json

from nomad.datamodel import EntryArchive
from nomad.parsing import MatchingParser

from nomad_bayesian_optimization.campaign_converter import (
    campaign_dict_to_schema_dict,
    discrete_candidate_count,
    measured_records,
    recommended_records,
)
from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
)
from nomad_bayesian_optimization.step_schema import (
    attach_step_package,
    derive_step_fields,
    make_step_instance,
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

        # The Bayesian optimization action injects a top-level ``status`` key and,
        # when available, ``step_field_meta`` (type/unit/description resolved from
        # the measurement schema) when it persists a campaign; use them if present
        # (a plain serialized campaign has neither and falls back to defaults).
        status = campaign.get('status')
        field_meta = campaign.get('step_field_meta')

        schema_dict = campaign_dict_to_schema_dict(campaign, status=status)
        archive.data = BayesianOptimization.m_from_dict(schema_dict)

        # Generate a typed per-campaign step schema (a subclass of ``Step``) from
        # the declared variables/targets, store it under ``archive.definitions``,
        # and record each measurement/recommendation as a typed step instance.
        field_specs = derive_step_fields(campaign, field_meta)
        step_def = attach_step_package(archive, field_specs)

        # Decoding the serialized dataframes requires BayBE (and pandas) to be
        # importable. Import lazily so that the parser entry point can still be
        # loaded in environments without the (heavy) BayBE stack.
        try:
            measured = measured_records(campaign)
            recommended = recommended_records(campaign)
            archive.data.n_candidates = discrete_candidate_count(campaign)
        except ImportError as exc:
            if logger is not None:
                logger.error(
                    'Could not decode BayBE campaign measurements because BayBE '
                    'is not installed. Install the "parsing" extra of this plugin.',
                    exc_info=exc,
                )
            return

        for record in measured:
            archive.data.steps.append(
                make_step_instance(step_def, record, field_specs, recommended=False)
            )
        for record in recommended:
            archive.data.steps.append(
                make_step_instance(step_def, record, field_specs, recommended=True)
            )
