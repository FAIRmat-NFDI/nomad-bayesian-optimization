import numpy as np
from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter
from baybe.recommenders import NaiveHybridSpaceRecommender, TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from temporalio import activity

from nomad_bayesian_optimization.actions.bayesian_optimization.models import (
    BayesianOptimizationInput,
)
from nomad_bayesian_optimization.schema_packages.cvd import CVD


@activity.defn
async def inference(data: BayesianOptimizationInput):
    """Perform Bayesian optimization based on the provided input data."""

    # Define optimization search space
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

    # Define optimization objective
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

    # Define acquisition function and recommender
    recommender = TwoPhaseMetaRecommender(recommender=NaiveHybridSpaceRecommender())

    # Get the samples from the given upload
    get_samples()

    # Start the optimization campaign
    campaign = Campaign(searchspace, objective, recommender)
    result = 0
    threshold = 0.05
    while abs(refractive_index_target - result) > threshold:
        df = campaign.recommend(batch_size=1)
        print('New recommendation:')
        print(df)
        print('Start testing recommendation...')
        archive = get_samples(df)
        result = archive.refractive_index
        print(f'Testing finished, refractive_index: {result}')
        df['refractive_index'] = [result]
        campaign.add_measurements(df)
    print('Optimization finished!')

    # At the end of the run, serialize the BayBE campaign in the given upload
    campaign.to_json()


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
        cvd_experiment = CVD().m_from_dict(row.to_dict())
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
        return cvd_experiment
        return cvd_experiment
