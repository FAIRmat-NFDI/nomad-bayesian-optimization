from nomad_bayesian_optimization.schema_packages.cvd import CVD
from temporalio import activity
from nomad.processing.data import Upload
from nomad.datamodel import ServerContext

from nomad_bayesian_optimization.actions.batch_cvd.models import (
    BatchCVDInput,
)


def get_randomized_parameters(substrate: str) -> dict:
    """
    Generates randomized process parameters for CVD based on the substrate.
    """
    import random

    if substrate == "Silicon":
        gas_flow_rate = random.uniform(0.1, 0.5)  # liter/min
        temperature = random.uniform(573, 1073)  # in Kelvin
        refractive_index = random.uniform(3.4, 3.6)  # Refractive index of Si is ~3.5
    elif substrate == "Silicon carbide":
        gas_flow_rate = random.uniform(0.2, 0.8)  # liter/min
        temperature = random.uniform(1073, 1873)  # in Kelvin
        refractive_index = random.uniform(2.5, 2.7)
    elif substrate == "Gallium nitride":
        gas_flow_rate = random.uniform(0.2, 0.7)  # liter/min
        temperature = random.uniform(1173, 1373)  # in Kelvin
        refractive_index = random.uniform(2.3, 2.5)
    else:  # Default case, though substrate is a Literal
        gas_flow_rate = random.uniform(0.1, 0.4)  # liter/min
        temperature = random.uniform(473, 873)  # in Kelvin
        refractive_index = random.uniform(1.4, 1.55)

    return {
        "gas_flow_rate": gas_flow_rate,
        "temperature": temperature,
        "refractive_index": refractive_index,
    }


@activity.defn
def batch_cvd(data: BatchCVDInput):
    """Creates a batch of CVD entries based on the provided input data."""
    upload = Upload.get(data.upload_id)
    user_id = data.user_id

    # Determine if user is authorized to get the upload.
    is_coauthor = isinstance(upload.coauthors, list) and user_id in upload.coauthors
    is_authorized = upload.main_author == user_id or is_coauthor

    # Raise error if not authorized
    if not is_authorized:
        raise PermissionError(
            f"User {user_id} is not authorized to access upload {data.upload_id}."
        )

    context = ServerContext(upload)
    results = {}

    for n in range(data.n_entries):
        params = get_randomized_parameters(data.substrate)
        entry_data = CVD(
            operator=data.operator,
            substrate=data.substrate,
            gas_flow_rate=params["gas_flow_rate"],
            temperature=params["temperature"],
            refractive_index=params["refractive_index"],
        )
        entry_data_dict = entry_data.m_to_dict(with_root_def=True)
        entry_path = f"entry_{n}.archive.json"
        with context.update_entry(entry_path, write=True, process=True) as archive:
            archive["data"] = entry_data_dict

        results[entry_path] = entry_data_dict

    return results
