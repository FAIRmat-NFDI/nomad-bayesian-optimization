"""Tests for reconciling measurements into a campaign without duplicating rows."""

import pandas as pd

from nomad_bayesian_optimization.campaign_builder import _record_key, seed_campaign


class _FakeCampaign:
    """Minimal stand-in for a BayBE campaign for testing ``seed_campaign``."""

    def __init__(self):
        self._measurements = pd.DataFrame()

    @property
    def measurements(self):
        return self._measurements

    def add_measurements(self, df):
        self._measurements = pd.concat([self._measurements, df], ignore_index=True)


def test_record_key_is_order_independent_and_rounds_floats():
    assert _record_key({'a': 1.0, 'b': 'x'}) == _record_key({'b': 'x', 'a': 1.0})
    # Floating-point noise below the rounding threshold collapses to one key.
    assert _record_key({'a': 1.0}) == _record_key({'a': 1.0 + 1e-12})


def test_seed_campaign_dedups():
    campaign = _FakeCampaign()
    records = [
        {'mass': 1.0, 'yield': 60.0},
        {'mass': 2.0, 'yield': 75.0},
    ]

    # First seed adds both.
    added = seed_campaign(campaign, records)
    assert added == 2
    assert len(campaign.measurements) == 2

    # Re-seeding the same records adds nothing (already present).
    added = seed_campaign(campaign, records)
    assert added == 0
    assert len(campaign.measurements) == 2

    # A genuinely new record is added.
    added = seed_campaign(campaign, [*records, {'mass': 3.0, 'yield': 82.0}])
    assert added == 1
    assert len(campaign.measurements) == 3


def test_seed_campaign_dedups_within_incoming_records():
    campaign = _FakeCampaign()
    added = seed_campaign(
        campaign,
        [{'mass': 1.0, 'yield': 60.0}, {'mass': 1.0, 'yield': 60.0}],
    )
    assert added == 1
    assert len(campaign.measurements) == 1


def test_seed_campaign_empty_records():
    campaign = _FakeCampaign()
    assert seed_campaign(campaign, []) == 0
    assert campaign.measurements.empty
