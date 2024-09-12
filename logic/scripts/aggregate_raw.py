
from pathlib import Path

import pandas as pd


path_to_raw_dir = "../../state/new_physics/data/raw"
path_to_output_dir = "../../state/new_physics/data/processed"
output_file_name = "aggregated_raw.pkl"
columns_to_save = ["q_squared", "costheta_mu", "costheta_K", "chi"]


def get_file_dc9(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    return dc9


path_to_raw_dir = Path(path_to_raw_dir)
path_to_output_dir = Path(path_to_output_dir)
path_to_output_file = path_to_output_dir.joinpath(output_file_name)

raw_data_filepaths = list(path_to_raw_dir.glob("*.pkl"))
raw_data = [pd.read_pickle(path)[columns_to_save] for path in raw_data_filepaths]
dc9_values = [get_file_dc9(path) for path in raw_data_filepaths]

raw_labeled_data = [df.assign(dc9=dc9) for df, dc9 in zip(raw_data, dc9_values)]

aggregate_data = pd.concat(raw_labeled_data)

pd.to_pickle(aggregate_data, path_to_output_file)
