
from helpers.data.datasets.histogram_image import Histogram_Image_Dataset

dset = Histogram_Image_Dataset()

n = 24_000
m = 100
num_bins = 12
agg_data_dir = "../../state/new_physics/data/processed"
save_dir = "../../state/new_physics/data/processed"

for level in {"gen", "det"}:
    for split in {"train", "eval"}:     
        dset.generate(
            level,
            split,
            n, m,
            num_bins,
            agg_data_dir,
            save_dir
        )

