
from library.data.datasets.aggregated_raw import Aggregated_Raw_Dataset

dset = Aggregated_Raw_Dataset()

columns = ["q_squared", "costheta_mu", "costheta_K", "chi"]

for level in {"gen", "det"}:
    for split in {"train", "eval"}: 
        dset.generate(
            level,
            split,
            columns,
            "../../state/new_physics/data/raw/signal",
            "../../state/new_physics/data/processed",
            sample=None
        )

