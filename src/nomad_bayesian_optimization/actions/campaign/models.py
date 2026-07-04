"""Pydantic models for the generic Bayesian optimization action.

These models define both the public action input (whose JSON schema drives the
auto-generated GUI form) and the internal payloads exchanged between the Temporal
workflow and its activities.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VariableSpec(BaseModel):
    """A single optimization variable (one dimension of the search space)."""

    name: str = Field(
        ...,
        description='Name of the variable. Used as the BayBE parameter name and as '
        'the measurement dataframe column.',
    )
    quantity: str | None = Field(
        None,
        description='Dot-separated path of the quantity within the target schema '
        "``data`` section to read this variable from. Defaults to ``name``.",
    )
    kind: Literal['continuous', 'numerical_discrete', 'categorical', 'substance'] = (
        Field(..., description='The type of variable.')
    )
    lower_bound: float | None = Field(
        None, description='Lower bound (continuous variables).'
    )
    upper_bound: float | None = Field(
        None, description='Upper bound (continuous variables).'
    )
    values: list[str] | None = Field(
        None,
        description='Allowed values for discrete/categorical variables. Numerical '
        'discrete values are given as strings and cast to float. Substance values '
        'use the ``name=SMILES`` form.',
    )
    tolerance: float | None = Field(
        None, description='Tolerance for numerical discrete variables.'
    )
    encoding: str | None = Field(
        None,
        description='Encoding for categorical (e.g. OHE, INT) or substance (e.g. '
        'MORDRED, RDKIT) variables.',
    )


class TargetSpec(BaseModel):
    """A single optimization target."""

    name: str = Field(
        ...,
        description='Name of the target. Used as the BayBE target name and as the '
        'measurement dataframe column.',
    )
    quantity: str | None = Field(
        None,
        description='Dot-separated path of the quantity within the target schema '
        "``data`` section to read this target from. Defaults to ``name``.",
    )
    mode: Literal['MAX', 'MIN', 'MATCH'] = Field(
        'MAX', description='Whether to maximize, minimize or match a target value.'
    )
    match_value: float | None = Field(
        None, description='Value to match (MATCH mode).'
    )
    sigma: float | None = Field(
        None, description='Width of the bell transformation (MATCH mode).'
    )
    weight: float | None = Field(
        None,
        description='Relative weight of this target within a multi-target '
        '(desirability) objective.',
    )


class BayesianOptimizationInput(BaseModel):
    """Public input for the Bayesian optimization action."""

    upload_id: str = Field(
        ..., description='Identifier of the upload the campaign operates on.'
    )
    user_id: str = Field(
        ..., description='Identifier of the user who initiated the action.'
    )
    schema_name: str = Field(
        ...,
        description='Name (or fully qualified suffix) of the measurement schema '
        'whose entries hold the optimization data, e.g. ``MySample``.',
    )
    variables: list[VariableSpec] = Field(
        ..., description='The variables spanning the search space.'
    )
    targets: list[TargetSpec] = Field(
        ..., description='The optimization targets.'
    )
    batch_size: int = Field(
        1, description='Number of measurements to suggest per iteration.'
    )
    scalarizer: str | None = Field(
        None,
        description='Scalarizer used to combine multiple targets in a desirability '
        'objective (e.g. GEOM_MEAN, MEAN). Only used when more than one target is '
        'given.',
    )
    campaign_name: str = Field(
        'campaign',
        description='Base name of the persisted campaign file '
        '(``<campaign_name>.json``). Use distinct names to run several campaigns in '
        'the same upload.',
    )


class DecisionInput(BaseModel):
    """Signal payload for accepting, rejecting or finishing at a suggestion."""

    action: Literal['accept', 'reject', 'finish'] = Field(
        'accept', description='Decision on the current suggestion.'
    )


class ResultsInput(BaseModel):
    """Signal payload reporting the entry that holds the recorded measurement."""

    entry_id: str = Field(
        ...,
        description='Identifier of the entry in which the measurement results were '
        'recorded.',
    )


# --- Internal activity payloads -------------------------------------------------


class RecommendInput(BaseModel):
    """Input for the ``recommend_next`` activity."""

    campaign_json: str
    batch_size: int = 1
    pending: list[dict[str, Any]] = Field(default_factory=list)


class RecommendOutput(BaseModel):
    """Output of the ``recommend_next`` activity."""

    campaign_json: str
    records: list[dict[str, Any]]


class AddMeasurementInput(BaseModel):
    """Input for the ``read_and_add_measurement`` activity."""

    campaign_json: str
    entry_id: str
    input: BayesianOptimizationInput


class PersistInput(BaseModel):
    """Input for the ``persist_campaign`` activity."""

    campaign_json: str
    upload_id: str
    campaign_name: str = 'campaign'
    status: str | None = None
