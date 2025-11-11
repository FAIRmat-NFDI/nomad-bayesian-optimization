from pydantic import BaseModel, Field


class BayesianOptimizationInput(BaseModel):
    """Base input model for bayesian optimization actions."""

    upload_id: str = Field(
        ...,
        description='Unique identifier for the upload associated with the workflow.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the workflow.'
    )
