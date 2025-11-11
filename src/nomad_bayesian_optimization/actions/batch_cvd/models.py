from typing import Literal

from pydantic import BaseModel, Field


class BatchCVDInput(BaseModel):
    """Base input model for creating a batch of CVD entries."""

    upload_id: str = Field(
        ...,
        description="Unique identifier for the upload associated with the workflow.",
    )
    user_id: str = Field(
        ..., description="Unique identifier for the user who initiated the workflow."
    )
    n_entries: int = Field(
        ...,
        description="Number of CVD entries to create in the batch.",
        gt=0,
    )
    operator: str = Field(..., description="Device operator.")

    substrate: Literal["Silicon carbide", "Silicon", "Gallium nitride"] = Field(
        ..., description="The used substrate material."
    )
