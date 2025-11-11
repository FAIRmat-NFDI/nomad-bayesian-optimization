from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_bayesian_optimization.actions.bayesian_optimization.activities import (
        inference,
    )
    from nomad_bayesian_optimization.actions.bayesian_optimization.models import (
        BayesianOptimizationInput,
    )


@workflow.defn
class BayesianOptimizationWorkflow:
    @workflow.run
    async def run(self, data: BayesianOptimizationInput) -> dict:
        retry_policy = RetryPolicy(
            maximum_attempts=3,
        )
        result = await workflow.execute_activity(
            inference,
            data,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )
        return result
