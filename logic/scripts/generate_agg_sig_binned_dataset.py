
from library.data.datasets.aggregated_signal_binned import Aggregated_Signal_Binned_Dataset

dset = Aggregated_Signal_Binned_Dataset()

features = ["q_squared", "costheta_mu", "costheta_K", "chi"]

for level in {"gen", "det"}:
    for split in {"train", "eval"}: 
        dset.generate(
            level,
            split,
            features,
            "../../state/new_physics/data/raw/signal",
            "../../state/new_physics/data/processed"
        )

