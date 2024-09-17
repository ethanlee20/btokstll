
from helpers.data.datasets.aggregated_raw import Aggregated_Raw_Dataset

dset = Aggregated_Raw_Dataset()

columns = ["q_squared", "costheta_mu", "costheta_K", "chi"]

for split in {"train", "eval"}: 
    dset.generate(
        columns,
        split,
        "../../state/new_physics/data/raw",
        "../../state/new_physics/data/processed"
    )

