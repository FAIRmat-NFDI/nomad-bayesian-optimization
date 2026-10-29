"""Pydantic models for the spin coating Bayesian optimization action.

The action is a specialization of the generic Bayesian optimization action with a
fixed search space and target. Its input only asks for what varies between
campaigns and is translated into a ``BayesianOptimizationInput``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nomad_bayesian_optimization.actions.campaign.models import (
    BayesianOptimizationInput,
    TargetSpec,
    VariableSpec,
)

SCHEMA_NAME = 'nomad_bayesian_optimization.schema_packages.spin_coating.SpinCoating'

# Search space bounds in the units of the schema quantities, and the decimals
# suggestions are rounded to so they can be set in the lab (0.01 M, 10 rpm, 1 s).
# These mirror ``scripts/generate_spin_coating_measurements.py``.
CONCENTRATION_BOUNDS = (0.2, 1.5)  # M
SPEED_BOUNDS = (1000.0, 6000.0)  # rpm
TIME_BOUNDS = (10.0, 60.0)  # s
CONCENTRATION_DECIMALS = 2
SPEED_DECIMALS = -1
TIME_DECIMALS = 0

# Width of the bell transformation as a fraction of the target thickness.
RELATIVE_SIGMA = 0.1


class SpinCoatingOptimizationInput(BaseModel):
    """Public input for the spin coating Bayesian optimization action."""

    upload_id: str = Field(
        ..., description='Identifier of the upload the campaign operates on.'
    )
    user_id: str = Field(
        ..., description='Identifier of the user who initiated the action.'
    )
    target_film_thickness: float = Field(
        ..., gt=0, description='Film thickness to achieve, in nanometers.'
    )
    batch_size: int = Field(
        1,
        ge=1,
        description='Number of steps, i.e. measurements to suggest per iteration.',
    )
    campaign_name: str = Field(
        'campaign_spin_coating',
        description='Base name of the persisted campaign file '
        '(``<campaign_name>.json``). Use distinct names to run several campaigns in '
        'the same upload.',
    )

    def to_bayesian_optimization_input(self) -> BayesianOptimizationInput:
        """Return the generic action input for this spin coating campaign."""
        variables = {
            'precursor_concentration': (CONCENTRATION_BOUNDS, CONCENTRATION_DECIMALS),
            'spin_coating_speed': (SPEED_BOUNDS, SPEED_DECIMALS),
            'spinning_time': (TIME_BOUNDS, TIME_DECIMALS),
        }
        return BayesianOptimizationInput(
            upload_id=self.upload_id,
            user_id=self.user_id,
            schema_name=SCHEMA_NAME,
            variables=[
                VariableSpec(
                    name=name,
                    kind='continuous',
                    lower_bound=lower,
                    upper_bound=upper,
                    decimals=decimals,
                )
                for name, ((lower, upper), decimals) in variables.items()
            ],
            targets=[
                TargetSpec(
                    name='film_thickness',
                    mode='MATCH',
                    match_value=self.target_film_thickness,
                    sigma=RELATIVE_SIGMA * self.target_film_thickness,
                )
            ],
            batch_size=self.batch_size,
            campaign_name=self.campaign_name,
        )
