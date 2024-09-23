
from library.data.datasets.bootstrapped_sets import Bootstrapped_Sets_Dataset

dset = Bootstrapped_Sets_Dataset()

label = "dc9"
n = 24_000
m = 1
agg_data_dir = "../../state/new_physics/data/processed"
save_dir = "../../state/new_physics/data/processed"

for level in {"gen", "det"}:
    for split in {"train", "eval"}:     
        dset.generate(
            level,
            split,
            label,
            n, m,
            agg_data_dir,
            save_dir
        )

