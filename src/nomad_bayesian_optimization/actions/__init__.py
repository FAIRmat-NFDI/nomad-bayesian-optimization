from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class BayesianOptimizationActionEntryPoint(ActionEntryPoint):
    """
    Action for triggering Bayesian optimization workflows.
    """

    def load(self):
        from nomad.actions import Action

        from nomad_bayesian_optimization.actions.campaign.activities import (
            build_campaign,
            persist_campaign,
            read_and_add_measurement,
            recommend_next,
            resolve_step_field_meta,
        )
        from nomad_bayesian_optimization.actions.campaign.workflows import (
            BayesianOptimizationWorkflow,
        )

        return Action(
            task_queue=self.task_queue,
            workflow=BayesianOptimizationWorkflow,
            activities=[
                build_campaign,
                resolve_step_field_meta,
                recommend_next,
                read_and_add_measurement,
                persist_campaign,
            ],
        )


bayesian_optimization = BayesianOptimizationActionEntryPoint(
    name='Bayesian Optimization',
    description=(
        'Runs a human-in-the-loop Bayesian optimization campaign over an upload: '
        'seeds a BayBE campaign from existing measurement entries, suggests new '
        'measurement parameters, and iterates as the user accepts suggestions and '
        'reports the resulting entries.'
    ),
)
