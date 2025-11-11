from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_bayesian_optimization.actions.bayesian_optimization.activities import (
        inference,
        write_campaign_to_schema,
    )
    from nomad_bayesian_optimization.actions.bayesian_optimization.models import (
        BayesianOptimizationInput,
        CreateBayesianOptimizationEntryInput,
    )


@workflow.defn
class BayesianOptimizationWorkflow:
    @workflow.run
    async def run(self, data: BayesianOptimizationInput) -> str:
        retry_policy = RetryPolicy(
            maximum_attempts=3,
        )
        campaign_json = await workflow.execute_activity(
            inference,
            data,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )
        entry_data = CreateBayesianOptimizationEntryInput(
            upload_id=data.upload_id, campaign_json=campaign_json
        )
        result = await workflow.execute_activity(
            write_campaign_to_schema,
            entry_data,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        return result
