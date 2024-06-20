api_url = ''
schema_name = 'nomad_bayesian_optimization.schema_packages.experiments.CVDExperiment'

# ======================================================================================
# Define the seach space
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter
from baybe.searchspace import SearchSpace

parameters = [
    CategoricalParameter(
        name='substrate',
        values=['Silicon carbide', 'Silicon', 'Gallium nitride'],
        encoding='OHE',  # one-hot encoding of categories
    ),
    NumericalContinuousParameter(
        name='temperature',
        bounds=(300, 600),
    ),
    NumericalContinuousParameter(
        name='gas_flow_rate',
        bounds=(0.2, 5),
    ),
]

searchspace = SearchSpace.from_product(parameters)

# ======================================================================================
# Define the optimization target
from baybe.objectives import SingleTargetObjective
from baybe.targets import NumericalTarget

refractive_index_target = 2.6473
refractive_index_sigma = 0.2
target = NumericalTarget(
    name='refractive_index',
    mode='MATCH',
    bounds=(
        refractive_index_target - refractive_index_sigma,
        refractive_index_target + refractive_index_sigma,
    ),
    transformation='BELL',
)
objective = SingleTargetObjective(target=target)

# ======================================================================================
# Define the acquisition strategy
from baybe.recommenders import SequentialGreedyRecommender, TwoPhaseMetaRecommender

recommender = TwoPhaseMetaRecommender(
    recommender=SequentialGreedyRecommender(
        hybrid_sampler='Farthest', sampling_percentage=0.3
    ),
)

# ======================================================================================
# Prepare upload to store the generated data

# ======================================================================================
# Start optimization loop
import time

import numpy as np
from baybe import Campaign

from nomad_bayesian_optimization.schema_packages.experiments import CVDExperiment

campaign = Campaign(searchspace, objective, recommender)


def get_samples(recommendations):
    """In this function you can decide how the actual experiment/simulation is
    performed. There are several alternatives:

    - Maybe you can control measurement devices directly through API calls.
    - Maybe you create a loop that waits until someone manually inserts the
      experiment results into NOMAD, and then query the results from it using
      the NOMAD API.
    - Maybe you run a simulation in this notebook
    - Maybe you run a simulation using an HPC batch system

    In this example, we will create entries by sampling from a
    fake model.
    """
    for _, row in recommendations.iterrows():
        cvd_experiment = CVDExperiment().m_from_dict(row.to_dict())
        temp_mu = 400
        temp_sigma = 200
        gas_flow_mu = 2
        gas_flow_sigma = 3
        ideal_substrate = 'Silicon carbide'
        refractive_index = float(
            2.6473
            * np.exp(-((cvd_experiment.temperature.m - temp_mu) ** 2 / temp_sigma**2))
            * np.exp(
                -(
                    (cvd_experiment.gas_flow_rate.m - gas_flow_mu) ** 2
                    / gas_flow_sigma**2
                )
            )
        )
        if cvd_experiment.substrate != ideal_substrate:
            refractive_index *= 0.9
        cvd_experiment.refractive_index = refractive_index
        time.sleep(2)
        return cvd_experiment


i = 0
result = 0
threshold = 0.1
while abs(refractive_index_target - result) > threshold:
    df = campaign.recommend(batch_size=1)
    print('New recommendation:')
    print(df)
    print('Testing recommendation...')
    archive = get_samples(df)
    print('Testing finished!')
    result = archive.refractive_index
    df['refractive_index'] = [result]
    campaign.add_measurements(df)
    i += 1

# At the end of the run, lets store the whole optimization run into an entry
import json

from nomad.datamodel import EntryArchive

from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
    BayesianOptimization,
)

archive = EntryArchive()
bopt = BayesianOptimization.from_baybe_campaign(campaign)
archive.data = bopt
bopt.normalize(archive, None)

with open('archive.json', 'w') as fout:
    json.dump(archive.m_to_dict(), fout, indent=2)

# print(bopt.search_space)
# campaign_dict = campaign.to_dict()
# recommendation = deserialize_dataframe(campaign_dict['_cached_recommendation'])
# measurements = deserialize_dataframe(campaign_dict['_measurements_exp'])
# print(measurements)
# with open("campaign.json", 'w') as fout:
#     json.dump(fout, campaign.to_dict())
# print(f'Optimal refractive index {result} found after {i} loops')
