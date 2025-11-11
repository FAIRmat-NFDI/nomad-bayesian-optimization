from temporalio import activity

from nomad_bayesian_optimization.actions.batch_cvd.models import BatchCVDInput


@activity.defn
async def batch_cvd(data: BatchCVDInput):
    """Creates a batch of CVD entries based on the provided input data."""
    pass
