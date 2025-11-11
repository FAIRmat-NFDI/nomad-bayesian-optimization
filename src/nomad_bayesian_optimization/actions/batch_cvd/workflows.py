from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_bayesian_optimization.actions.batch_cvd.activities import batch_cvd
    from nomad_bayesian_optimization.actions.batch_cvd.models import BatchCVDInput


@workflow.defn
class BatchCVDWorkflow:
    @workflow.run
    async def run(self, data: BatchCVDInput) -> dict:
        retry_policy = RetryPolicy(
            maximum_attempts=3,
        )
        result = await workflow.execute_activity(
            batch_cvd,
            data,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        return result
