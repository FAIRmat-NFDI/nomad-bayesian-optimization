from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad.actions.manager import request_signal_input

    from nomad_bayesian_optimization.actions.campaign.activities import (
        build_campaign,
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
        ResultsInput,
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


@workflow.defn
class BayesianOptimizationWorkflow:
    def __init__(self) -> None:
        self._decision: DecisionInput | None = None
        self._results: ResultsInput | None = None

    @workflow.signal
    def decision(self, data: DecisionInput) -> None:
        """Accept, reject or finish the current suggestion."""
        self._decision = data

    @workflow.signal
    def results(self, data: ResultsInput) -> None:
        """Report the entry holding the recorded measurement."""
        self._results = data

    @workflow.run
    async def run(self, data: BayesianOptimizationInput) -> str:
        retry_policy = RetryPolicy(maximum_attempts=3)
        action_instance_id = workflow.info().workflow_id

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

        pending: list[dict] = []
        n_measurements = 0

        while True:
            recommendation = await workflow.execute_activity(
                recommend_next,
                RecommendInput(
                    campaign_json=campaign_json,
                    batch_size=data.batch_size,
                    pending=pending,
                ),
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=retry_policy,
            )
            campaign_json = recommendation.campaign_json
            records = recommendation.records

            await persist(campaign_json, 'Suggesting')

            # Ask the user to accept / reject / finish this suggestion.
            self._decision = None
            await request_signal_input(
                action_instance_id=action_instance_id,
                user_id=data.user_id,
                signal_fn_name='decision',
                title='New measurement suggestion',
                content=_format_records(records),
                initial_data={'action': 'accept'},
                timeout=SIGNAL_REQUEST_TIMEOUT,
            )
            await workflow.wait_condition(lambda: self._decision is not None)
            decision = self._decision

            if decision.action == 'finish':
                break
            if decision.action == 'reject':
                pending.extend(records)
                continue

            # Accepted: wait for the user to perform the measurement and report the
            # entry that holds the results. This wait may span days.
            pending = []
            await persist(campaign_json, 'Acquiring')

            self._results = None
            await request_signal_input(
                action_instance_id=action_instance_id,
                user_id=data.user_id,
                signal_fn_name='results',
                title='Report your measurement',
                content=(
                    'Perform the suggested measurement, record it in a new entry of '
                    'the target schema, and submit that entry_id.'
                ),
                initial_data={'entry_id': ''},
                timeout=SIGNAL_REQUEST_TIMEOUT,
            )
            await workflow.wait_condition(lambda: self._results is not None)
            results = self._results

            campaign_json = await workflow.execute_activity(
                read_and_add_measurement,
                AddMeasurementInput(
                    campaign_json=campaign_json,
                    entry_id=results.entry_id,
                    input=data,
                ),
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=retry_policy,
            )
            n_measurements += 1

        await persist(campaign_json, 'Finished')
        return (
            f'Campaign finished after {n_measurements} recorded '
            f'measurement(s).'
        )
