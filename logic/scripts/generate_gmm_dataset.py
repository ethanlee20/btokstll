
from library.data.datasets.gmm import Gaussian_Mixture_Model_Dataset

dset = Gaussian_Mixture_Model_Dataset()

save_dir = "../../state/new_physics/data/processed"

for level in ["gen", "det"]:
    for split in ["train", "eval", "lin_eval"]:     
        dset.generate(
            level, split, save_dir
        )

