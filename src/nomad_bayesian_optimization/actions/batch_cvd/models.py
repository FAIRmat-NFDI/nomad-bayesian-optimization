from enum import Enum

from pydantic import BaseModel, Field


class Device(Enum):
    DEVICE_1 = 'device_1'
    DEVICE_2 = 'device_2'
    DEVICE_3 = 'device_3'


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
    operator: str = Field(
        ..., description='Name of the operator performing the batch creation.'
    )

    device: Device = Field(
        ..., description='Device on which the batch creation is performed.'
    )
