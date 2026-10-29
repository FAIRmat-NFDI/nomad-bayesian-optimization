from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad.actions.manager import request_signal_input

    from nomad_bayesian_optimization.actions.campaign.activities import (
        build_campaign,
        clear_recommendation,
        persist_campaign,
        read_and_add_measurement,
        recommend_next,
        resolve_step_field_meta,
    )
    from nomad_bayesian_optimization.actions.campaign.models import (
        AddMeasurementInput,
        BayesianOptimizationInput,
        DecisionInput,
        PersistInput,
        RecommendInput,
    )

# Activities that touch NOMAD/BayBE can be slow; give them a generous timeout.
ACTIVITY_TIMEOUT = timedelta(minutes=10)
# The activity that *records* a pending signal request is quick; the actual wait
# for the human happens in ``workflow.wait_condition`` below.
SIGNAL_REQUEST_TIMEOUT = timedelta(minutes=1)


def _format_records(records: list[dict]) -> str:
    """Render recommended measurements as a markdown table for the GUI form."""
    if not records:
        return 'No recommendation could be produced.'
    columns = list(records[0].keys())
    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join('---' for _ in columns) + ' |'
    rows = [
        '| ' + ' | '.join(str(record.get(column, '')) for column in columns) + ' |'
        for record in records
    ]
    table = '\n'.join([header, separator, *rows])
    return f'Suggested next measurement(s):\n\n{table}'


def _decision_content(
    records: list[dict], schema_name: str, note: str | None = None
) -> str:
    """Render the suggestion together with instructions for the decision form."""
    schema = schema_name.rsplit('.', 1)[-1]
    instructions = (
        f'Perform the suggested measurement(s) and record each in a new `{schema}` '
        'entry, then **accept** and list the entry IDs. **Reject** to get a '
        'different suggestion, or **finish** to end the campaign (any listed entry '
        'IDs are recorded first).'
    )
    parts = [_format_records(records), instructions]
    if note:
        parts.insert(0, f'**{note}**')
    return '\n\n'.join(parts)


@workflow.defn
class BayesianOptimizationWorkflow:
    def __init__(self) -> None:
        self._decision: DecisionInput | None = None

    @workflow.signal
    def decision(self, data: DecisionInput) -> None:
        """Accept (reporting the measured entries), reject or finish the current
        suggestion."""
        self._decision = data

    async def _request_decision(
        self,
        data: BayesianOptimizationInput,
        records: list[dict],
        note: str | None = None,
    ) -> DecisionInput:
        """Ask the user to decide on the suggestion and wait for the answer. This
        wait may span days, as accepting requires performing the measurement."""
        self._decision = None
        await request_signal_input(
            action_instance_id=workflow.info().workflow_id,
            user_id=data.user_id,
            signal_fn_name='decision',
            title='New measurement suggestion',
            content=_decision_content(records, data.schema_name, note),
            initial_data={'action': 'accept', 'entry_ids': []},
            timeout=SIGNAL_REQUEST_TIMEOUT,
        )
        await workflow.wait_condition(lambda: self._decision is not None)
        return self._decision

    @workflow.run
    async def run(self, data: BayesianOptimizationInput) -> str:
        return await self._run_campaign(data)

    async def _run_campaign(self, data: BayesianOptimizationInput) -> str:
        """Run the campaign loop. Shared with specialized workflows that subclass
        this one and translate their own input into a ``BayesianOptimizationInput``.
        """
        retry_policy = RetryPolicy(maximum_attempts=3)

        # Resolve the step-field metadata (type/unit/description) from the
        # measurement schema once; it is injected into every persisted campaign so
        # the parser can generate a typed step schema.
        field_meta = await workflow.execute_activity(
            resolve_step_field_meta,
            data,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=retry_policy,
        )

        def persist(campaign_json: str, status: str):
            return workflow.execute_activity(
                persist_campaign,
                PersistInput(
                    campaign_json=campaign_json,
                    upload_id=data.upload_id,
                    campaign_name=data.campaign_name,
                    status=status,
                    field_meta=field_meta,
                ),
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=retry_policy,
            )

        campaign_json = await workflow.execute_activity(
            build_campaign,
            data,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=retry_policy,
        )

        decimals = {
            variable.name: variable.decimals
            for variable in data.variables
            if variable.decimals is not None
        }
        pending: list[dict] = []
        n_measurements = 0

        while True:
            recommendation = await workflow.execute_activity(
                recommend_next,
                RecommendInput(
                    campaign_json=campaign_json,
                    batch_size=data.batch_size,
                    pending=pending,
                    decimals=decimals,
                ),
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=retry_policy,
            )
            campaign_json = recommendation.campaign_json
            records = recommendation.records

            await persist(campaign_json, 'Suggesting')

            # Ask the user to accept / reject / finish this suggestion. Accepting
            # reports the entries holding the measurements, so ask again when none
            # are given (a model validator would make Temporal drop the signal).
            decision = await self._request_decision(data, records)
            while decision.action == 'accept' and not decision.entry_ids:
                decision = await self._request_decision(
                    data,
                    records,
                    note='Accepting requires the entry ID(s) of the recorded '
                    'measurement(s).',
                )

            if decision.action == 'reject':
                pending.extend(records)
                continue

            pending = []
            if decision.entry_ids:
                campaign_json = await workflow.execute_activity(
                    read_and_add_measurement,
                    AddMeasurementInput(
                        campaign_json=campaign_json,
                        entry_ids=decision.entry_ids,
                        input=data,
                    ),
                    start_to_close_timeout=ACTIVITY_TIMEOUT,
                    retry_policy=retry_policy,
                )
                n_measurements += len(decision.entry_ids)

            if decision.action == 'finish':
                break

        # The last suggestion was never attempted; drop it so it does not show up
        # as a step of the finished campaign.
        campaign_json = await workflow.execute_activity(
            clear_recommendation,
            campaign_json,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=retry_policy,
        )
        await persist(campaign_json, 'Finished')
        return f'Campaign finished after {n_measurements} recorded measurement(s).'
