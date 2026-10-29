from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_bayesian_optimization.actions.campaign.workflows import (
        BayesianOptimizationWorkflow,
    )
    from nomad_bayesian_optimization.actions.spin_coating.models import (
        SpinCoatingOptimizationInput,
    )


@workflow.defn
class SpinCoatingOptimizationWorkflow(BayesianOptimizationWorkflow):
    """Bayesian optimization of spin coating parameters for a target film thickness.

    Runs the generic campaign loop, including its ``decision`` and ``results``
    signals, with a fixed search space and target.
    """

    @workflow.run
    async def run(self, data: SpinCoatingOptimizationInput) -> str:
        return await self._run_campaign(data.to_bayesian_optimization_input())
