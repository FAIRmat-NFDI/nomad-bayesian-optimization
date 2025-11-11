from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class BayesianOptimizationActionEntryPoint(ActionEntryPoint):
    """
    Action for triggering Bayesian optimization workflows.
    """

    def load(self):
        from nomad.actions import Action

        from nomad_bayesian_optimization.actions.bayesian_optimization.activities import (
            inference,
            write_campaign_to_schema,
        )
        from nomad_bayesian_optimization.actions.bayesian_optimization.workflows import (
            BayesianOptimizationWorkflow,
        )

        return Action(
            task_queue=self.task_queue,
            workflow=BayesianOptimizationWorkflow,
            activities=[inference, write_campaign_to_schema],
        )


class BatchCVDActionEntryPoint(ActionEntryPoint):
    """
    Action for creating a batch of CVD entries.
    """

    def load(self):
        from nomad.actions import Action

        from nomad_bayesian_optimization.actions.batch_cvd.activities import batch_cvd
        from nomad_bayesian_optimization.actions.batch_cvd.workflows import (
            BatchCVDWorkflow,
        )

        return Action(
            task_queue=self.task_queue,
            workflow=BatchCVDWorkflow,
            activities=[batch_cvd],
        )


bayesian_optimization = BayesianOptimizationActionEntryPoint(
    name='Bayesian Optimization',
    description='Used to trigger Bayesian optimization workflows.',
)


batch_cvd = BatchCVDActionEntryPoint(
    name='Batch CVD',
    description='Used to create a batch of CVD entries.',
)
