
from pathlib import Path

import numpy as np
#import sklearn.mixture
import pandas as pd
import torch
from torch.utils.data import Dataset

from .bootstrapped_sets import Bootstrapped_Sets_Dataset


def make_feature_file_name(level, split):
    assert level in {"gen", "det"}
    assert split in {"train", "eval", "lin_eval"}
    name = f"gmm_feat_{level}_{split}.npy"
    return name


def make_label_file_name(level, split):
    assert level in {"gen", "det"}
    assert split in {"train", "eval", "lin_eval"}
    name = f"gmm_label_{level}_{split}.npy"
    return name


def fit_gmm(df_set, gmm):
    x = df_set.to_numpy()

    gmm.fit(x)

    weights = gmm.weights_ # shape: n_components
    means = gmm.means_ # shape: n_components, n_features
    covs = gmm.covariances_ # shape: n_components, n_features, n_features (for full)

    covs = covs.reshape(-1, covs.shape[1]*covs.shape[2])
    weights = np.expand_dims(weights, axis=1)
    breakpoint()
    out = np.concat([weights, means, covs], axis=1)
    return out


class Gaussian_Mixture_Model_Dataset(Dataset):
    def __init__(self):
        pass

    def generate(
            self, level, split, save_dir
        ):
        save_dir = Path(save_dir)

        bootsets_dset = Bootstrapped_Sets_Dataset()
        bootsets_dset.load(level, split, save_dir)

        sets = bootsets_dset.sets

        n_feat = sets.loc[0].shape[1]
        gmm = sklearn.mixture.BayesianGaussianMixture(
            n_components=100,
            covariance_type="full",
            weight_concentration_prior=1e2,
            weight_concentration_prior_type="dirichlet_process",
            mean_precision_prior=1e-2,
            covariance_prior=1e0*np.eye(n_feat),
            init_params="kmeans",
            max_iter=3000,
            warm_start=True,
            verbose=2
        )

        gmm_fit_results = []
        for i in range(len(bootsets_dset)):
            df_set = sets.loc[i]
            gmm_fit = fit_gmm(df_set, gmm)
            gmm_fit = np.expand_dims(gmm_fit, axis=0)
            gmm_fit_results.append(gmm_fit)
        
        gmm_fit_results = np.concat(gmm_fit_results, axis=0)

        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        np.save(feature_file_path, gmm_fit_results)
        np.save(label_file_path, bootsets_dset.labels)
        
    def load(self, level, split, save_dir, device):
        save_dir = Path(save_dir)
        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        self.gmms = np.load(feature_file_path, allow_pickle=True)
        self.labels = np.load(label_file_path, allow_pickle=True)

        self.gmms = self.gmms.astype("float32")
        self.labels = self.labels.astype("float32")

        self.gmms = torch.from_numpy(self.gmms).to(device)
        self.labels = torch.from_numpy(self.labels).to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.gmms[index]
        y = self.labels[index]
        return x, y