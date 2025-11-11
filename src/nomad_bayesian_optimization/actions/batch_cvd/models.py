from enum import Enum

from pydantic import BaseModel, Field


class Substrate(Enum):
    SIC = 'Silicon carbide'
    SI = 'Silicon 2'
    GaN = 'Gallium nitride'


class BatchCVDInput(BaseModel):
    """Base input model for creating a batch of CVD entries."""

    upload_id: str = Field(
        ...,
        description='Unique identifier for the upload associated with the workflow.',
    )
    n_entries: int = Field(
        ...,
        description='Number of CVD entries to create in the batch.',
        gt=0,
    )
    operator: str = Field(..., description='Device operator.')

    substrate: Substrate = Field(..., description='The used substrate material.')
