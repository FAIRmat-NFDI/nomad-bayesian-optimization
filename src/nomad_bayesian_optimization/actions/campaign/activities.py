"""Temporal activities for the generic Bayesian optimization action.

All BayBE and NOMAD I/O happens here (never in the workflow, which must stay
deterministic). The BayBE campaign state is passed between activities as the JSON
string produced by ``campaign.to_json()`` and persisted, after every step, as a
raw ``<campaign_name>.json`` file in the upload. That file is picked up by the
BayBE parser, which (re)builds the ``BayesianOptimization`` entry.
"""

import json

import pandas as pd
from baybe import Campaign
from nomad.datamodel.context import ServerContext
from nomad.processing.data import Upload
from temporalio import activity

from nomad_bayesian_optimization.actions.campaign.models import (
    AddMeasurementInput,
    BayesianOptimizationInput,
    PersistInput,
    RecommendInput,
    RecommendOutput,
)
from nomad_bayesian_optimization.campaign_builder import (
    build_campaign as build_baybe_campaign,
)
from nomad_bayesian_optimization.measurement_reader import (
    check_authorized,
    read_measurement_records,
)


def _campaign_file(campaign_name: str) -> str:
    return f'{campaign_name}.json'


@activity.defn
async def build_campaign(data: BayesianOptimizationInput) -> str:
    """Build a campaign from the input, resuming from a persisted campaign if any.

    If ``<campaign_name>.json`` already exists in the upload, the campaign is
    restored from it (full state, including previously recorded measurements).
    Otherwise a fresh campaign is built from the variable/target specification and
    seeded with all existing measurements found in the upload.
    """
    upload = Upload.get(data.upload_id)
    check_authorized(upload, data.user_id)

    campaign_file = _campaign_file(data.campaign_name)
    if upload.upload_files.raw_path_exists(campaign_file):
        with upload.upload_files.raw_file(campaign_file, 'r') as f:
            payload = json.load(f)
        # The persisted file carries an injected ``status`` key that BayBE does not
        # expect; drop it before restoring the campaign.
        payload.pop('status', None)
        campaign = Campaign.from_json(json.dumps(payload))
        return campaign.to_json()

    campaign = build_baybe_campaign(data)
    records = read_measurement_records(
        upload, data.user_id, data.schema_name, data.variables, data.targets
    )
    if records:
        campaign.add_measurements(pd.DataFrame(records))
    return campaign.to_json()


@activity.defn
async def recommend_next(data: RecommendInput) -> RecommendOutput:
    """Recommend the next batch of measurements to try.

    Rejected recommendations are passed back as ``pending_experiments`` so BayBE
    avoids suggesting them again.
    """
    campaign = Campaign.from_json(data.campaign_json)

    pending_df = pd.DataFrame(data.pending) if data.pending else None
    if pending_df is not None and not pending_df.empty:
        recommendation = campaign.recommend(
            batch_size=data.batch_size, pending_experiments=pending_df
        )
    else:
        recommendation = campaign.recommend(batch_size=data.batch_size)

    records = json.loads(recommendation.to_json(orient='records'))
    return RecommendOutput(campaign_json=campaign.to_json(), records=records)


@activity.defn
async def read_and_add_measurement(data: AddMeasurementInput) -> str:
    """Read the reported results entry and add it to the campaign."""
    spec = data.input
    upload = Upload.get(spec.upload_id)
    records = read_measurement_records(
        upload,
        spec.user_id,
        spec.schema_name,
        spec.variables,
        spec.targets,
        entry_ids=[data.entry_id],
    )

    campaign = Campaign.from_json(data.campaign_json)
    if records:
        campaign.add_measurements(pd.DataFrame(records))
    return campaign.to_json()


@activity.defn
async def persist_campaign(data: PersistInput) -> str:
    """Persist the raw campaign JSON as a file in the upload.

    Writing the file with ``process=True`` triggers the BayBE parser, which
    regenerates the ``BayesianOptimization`` entry (steps, status, progress plot).
    A top-level ``status`` key is injected so the parser can record the campaign
    status on the entry.
    """
    upload = Upload.get(data.upload_id)
    context = ServerContext(upload)
    campaign_data = json.loads(data.campaign_json)

    campaign_file = _campaign_file(data.campaign_name)
    with context.update_entry(campaign_file, write=True, process=True) as content:
        content.clear()
        content.update(campaign_data)
        if data.status:
            content['status'] = data.status
    return campaign_file
