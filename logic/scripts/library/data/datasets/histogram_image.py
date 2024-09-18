
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

from .aggregated_raw import Aggregated_Raw_Dataset
from .bootstrap_sets import bootstrap_sets


def make_image_file_name(level, split):
    name = f"histogram_images_{level}_{split}.npy"
    return name


def make_label_file_name(level, split):
    name = f"histogram_labels_{level}_{split}.npy"
    return name


def normalize_hist_image(hist_image):
    means = np.mean(hist_image, axis=(1,2))
    means = np.expand_dims(means, axis=(1,2))
    stdevs = np.std(hist_image, axis=(1,2))
    stdevs = np.expand_dims(stdevs, axis=(1,2))
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

    def generate(self, level, split, n, m, num_bins, agg_data_dir, save_dir):

        save_dir = Path(save_dir)
        image_file_name = make_image_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        image_file_path = save_dir.joinpath(image_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        agg_dset = Aggregated_Raw_Dataset()
        agg_dset.load(split, agg_data_dir)

        label_column = "dc9"
        sampled_sets, labels = bootstrap_sets(agg_dset.data.loc[level], label_column, n, m)

        hist_images = [make_histogram_image(df, num_bins) for df in sampled_sets]
        hist_images = np.concatenate(
            [np.expand_dims(h, axis=0) for h in hist_images],
            axis=0
        )
        labels = np.array(labels)

        np.save(image_file_path, hist_images)
        np.save(label_file_path, labels)

    def load(self, level, split, save_dir, device):
        save_dir = Path(save_dir)
        image_file_name = make_image_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        image_file_path = save_dir.joinpath(image_file_name)
        label_file_path = save_dir.joinpath(label_file_name)
        images = np.load(image_file_path, allow_pickle=True)
        labels = np.load(label_file_path, allow_pickle=True)

        assert len(images) == len(labels)

        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        y = torch.unsqueeze(y, 0)
        return x, y