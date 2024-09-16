
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from .aggregated_raw import Aggregated_Raw_Dataset
from .bootstrap_sets import bootstrap_sets


def make_file_name(split):
    name = f"histogram_images_{split}.npy"
    return name


def normalize_hist_image(hist_image):
    means = np.mean(hist_image, axis=(1,2))
    stdevs = np.stdev(hist_image, axis=(1,2))
    hist_image = (hist_image - means) / stdevs
    return hist_image


def make_histogram_image(df, n_bins):

    make_bin_edges = lambda start, stop : np.linspace(start, stop, num=n_bins+1)

    bins = {
        "q_squared": make_bin_edges(0, 20),
        "costheta_mu": make_bin_edges(-1, 1),
        "costheta_K": make_bin_edges(-1, 1),
        "chi": make_bin_edges(0, 2*np.pi)
    }

    combinations = [
        ["q_squared", "costheta_mu"],
        ["q_squared", "costheta_K"],
        ["q_squared", "chi"],
        ["costheta_K", "chi"],
        ["costheta_mu", "chi"],
        ["costheta_K", "costheta_mu"],
        ["q_squared", "q_squared"],
        ["costheta_mu", "costheta_mu"],
        ["costheta_K", "costheta_K"],
        ["chi", "chi"],
    ]

    hists = [ 
        np.histogram2d(df[c[0]], df[c[1]], bins=(bins[c[0]], bins[c[1]]))[0]
        for c in combinations
    ]

    hist_image = np.concatenate(
        [np.expand_dims(h, axis=0) for h in hists],
        axis=0
    )

    hist_image = normalize_hist_image(hist_image)

    return hist_image


class Histogram_Image_Dataset(Dataset):
    def __init__(self):
        pass

    def generate(self, split, n, m, num_bins, agg_data_dir, save_dir):

        save_dir = Path(save_dir)
        save_file_name = make_file_name(split)
        save_path = save_dir.joinpath(save_file_name)

        agg_dset = Aggregated_Raw_Dataset()
        agg_dset.load(split, agg_data_dir)

        label = "dc9"
        sampled_sets = bootstrap_sets(agg_dset.data, label, n, m)

        hist_images = [make_histogram_image(df, num_bins) for df in sampled_sets]
        hist_images = np.concatenate(
            [np.expand_dims(h, axis=0) for h in hist_images],
            axis=0
        )
        
        np.save(hist_images, save_path)

    def load(self, split, save_dir):
        save_dir = Path(save_dir)
        save_file_name = make_file_name(split)
        save_path = save_dir.joinpath(save_file_name)
        ar = np.load(save_path, allow_pickle=True)
        self.data = ar