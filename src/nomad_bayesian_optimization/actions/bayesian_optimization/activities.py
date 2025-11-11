import glob
import json
import os

import numpy as np
import pandas as pd
from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter
from baybe.recommenders import NaiveHybridSpaceRecommender, TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from nomad.datamodel.context import ServerContext
from nomad.files import StagingUploadFiles
from nomad.processing.data import Upload
from temporalio import activity

from nomad_bayesian_optimization.actions.bayesian_optimization.models import (
    BayesianOptimizationInput,
    CreateBayesianOptimizationEntryInput,
)
from nomad_bayesian_optimization.schema_packages.cvd import CVD


def get_measurements_from_upload(upload_id: str, user_id: str) -> list:
    """
    Reads all archive files from an upload directory, extracts data from CVD schemas,
    and returns a list of measurements.
    """
    upload = Upload.get(upload_id)
    is_coauthor = isinstance(upload.coauthors, list) and user_id in upload.coauthors
    is_authorized = upload.main_author == user_id or is_coauthor

    if not is_authorized:
        raise PermissionError(
            f"User {user_id} is not authorized to access upload {upload_id}."
        )

    measurements = []
    upload_path = StagingUploadFiles(upload_id=upload_id).os_path
    archive_files = glob.glob(
        os.path.join(upload_path, "**", "*.archive.json"), recursive=True
    )

    for file_path in archive_files:
        try:
            with open(file_path, "r") as f:
                archive_json = json.load(f)

            if (
                "data" in archive_json
                and "m_def" in archive_json.get("data", {})
                and archive_json["data"]["m_def"].endswith("CVD")
            ):
                data = archive_json["data"]
                measurements.append(
                    {
                        "substrate": data["substrate"],
                        "temperature": data["temperature"]["value"],
                        "gas_flow_rate": data["gas_flow_rate"]["value"],
                        "refractive_index": data["refractive_index"],
                    }
                )
        except Exception:
            # File might be corrupted or not have the expected structure
            pass
    return measurements


@activity.defn
async def inference(data: BayesianOptimizationInput):
    """Perform Bayesian optimization based on the provided input data."""

    # Define optimization search space
    parameters = [
        CategoricalParameter(
            name="substrate",
            values=["Silicon carbide", "Silicon", "Gallium nitride"],
            encoding="OHE",  # one-hot encoding of categories
        ),
        NumericalContinuousParameter(
            name="temperature",
            bounds=(300, 600),
        ),
        NumericalContinuousParameter(
            name="gas_flow_rate",
            bounds=(0.2, 5),
        ),
    ]
    searchspace = SearchSpace.from_product(parameters)

    # Define optimization objective
    refractive_index_target = data.refractive_index_target
    refractive_index_sigma = 0.2
    target = NumericalTarget(
        name="refractive_index",
        mode="MATCH",
        bounds=(
            refractive_index_target - refractive_index_sigma,
            refractive_index_target + refractive_index_sigma,
        ),
        transformation="BELL",
    )
    objective = SingleTargetObjective(target=target)

    # Define acquisition function and recommender
    recommender = TwoPhaseMetaRecommender(recommender=NaiveHybridSpaceRecommender())

    # Start the optimization campaign
    campaign = Campaign(searchspace, objective, recommender)

    # Add existing measurements from the upload to the campaign
    measurements = get_measurements_from_upload(data.upload_id, data.user_id)

    if measurements:
        df_measurements = pd.DataFrame(measurements)
        campaign.add_measurements(df_measurements)

    result = 0
    threshold = 0.05
    while abs(refractive_index_target - result) > threshold:
        df = campaign.recommend(batch_size=1)
        print("New recommendation:")
        print(df)
        print("Start testing recommendation...")
        archive = get_samples(df, refractive_index_target)
        if archive:
            result = float(archive.refractive_index)
            print(f"Testing finished, refractive_index: {result}")
            df["refractive_index"] = [result]
        campaign.add_measurements(df)
    print("Optimization finished!")

    # At the end of the run, serialize the BayBE campaign in the given upload
    return campaign.to_json()


def get_samples(recommendations, refractive_index_target: float):
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
        ideal_substrate = "Silicon carbide"
        refractive_index = float(
            refractive_index_target
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


def decode_dataframe(encoded_str):
    """Decodes a base64, pickled pandas DataFrame."""
    import base64
    import pickle

    decoded_bytes = base64.b64decode(encoded_str)
    measurements_df = pickle.loads(decoded_bytes)
    return measurements_df


@activity.defn
async def write_campaign_to_schema(data: CreateBayesianOptimizationEntryInput) -> str:
    """
    Transforms a BayBE campaign JSON into a dictionary that conforms to the
    BayesianOptimization schema.
    """
    campaign_data = json.loads(data.campaign_json)

    # 1. Process parameters from searchspace
    parameters = []
    discrete_params = (
        campaign_data.get("searchspace", {}).get("discrete", {}).get("parameters", [])
    )
    for param in discrete_params:
        parameters.append(
            {
                "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.CategoricalParameter",
                "name": param["name"],
                "values": param["values"],
            }
        )

    continuous_params = (
        campaign_data.get("searchspace", {}).get("continuous", {}).get("parameters", [])
    )
    for param in continuous_params:
        parameters.append(
            {
                "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.ContinuousParameter",
                "name": param["name"],
                "lower_bound": param["bounds"]["lower"],
                "upper_bound": param["bounds"]["upper"],
            }
        )

    # 2. Process objective
    objective_data = campaign_data.get("objective", {})
    target_data = objective_data.get("target", {})
    transformation_data = target_data.get("transformation", {})
    center = transformation_data.get("center", 0)
    sigma = transformation_data.get("sigma", 0)

    objective = {
        "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.Objective",
        "type": objective_data.get("type"),
        "target": {
            "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.Target",
            "type": target_data.get("type"),
            "name": target_data.get("name"),
            "mode": target_data.get("mode"),
            "transformation": transformation_data.get("type"),
            "bounds": {
                "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.Bounds",
                "lower": center - sigma,
                "upper": center + sigma,
            },
        },
    }

    # 3. Process recommender (simplified version)
    recommender_data = campaign_data.get("recommender", {})
    recommender = {
        "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.Recommender",
        "type": recommender_data.get("type"),
    }

    # 4. Process steps from measurements
    steps = []
    measurements_exp_encoded = campaign_data.get("measurements_exp")
    if measurements_exp_encoded:
        measurements_df = decode_dataframe(measurements_exp_encoded)
        for _, row in measurements_df.iterrows():
            steps.append(
                {
                    "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.Step",
                    "values_used": row.to_dict(),
                }
            )

    # 5. Assemble the final dictionary
    bayesian_optimization_dict = {
        "m_def": "nomad_bayesian_optimization.schema_packages.bayesian_optimization.BayesianOptimization",
        "status": "Finished",
        "parameters": parameters,
        "objective": objective,
        "recommender": recommender,
        "steps": steps,
        "n_steps": len(steps),
    }
    upload = Upload.get(data.upload_id)
    context = ServerContext(upload)
    entry_path = "bayesian_optimization.archive.json"
    with context.update_entry(entry_path, write=True, process=True) as archive:
        archive["data"] = bayesian_optimization_dict

    return entry_path
