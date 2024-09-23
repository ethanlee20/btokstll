
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .aggregated_raw import Aggregated_Raw_Dataset

def make_feature_file_name(level, split):
    name = f"boot_sets_feat_{level}_{split}.pkl"
    return name


def make_label_file_name(level, split):
    name = f"boot_sets_label_{level}_{split}.npy"
    return name


def bootstrap_sets(df, label, n, m):
    """
    Bootstrap sets from the distribution of each label in a
    source dataframe.

    Bootstrapping refers to sampling with replacement.
    
    Parameters
    ----------
    df : pd.DataFrame
        The source dataframe to sample from
    label : str 
        Name of the column specifying the labels.
    n : int
        The number of elements per set.
    m : int
        The number of sets per label. 
    
    Returns
    -------
    sets : pd.DataFrame
        The sampled sets.
    labels : np.ndarray
        The corresponding labels.
    """
    
    df_grouped = df.groupby(label)
    
    sets = []
    labels = []

    for label_value, df_label in df_grouped:

        for i in range(m):
            
            df_set = df_label.sample(n=n, replace=True).drop(columns=label)
            sets.append(df_set)
            labels.append(label_value)

    sets = pd.concat(sets, keys=range(len(sets)), names=["set", "event"])
    labels = np.array(labels)

    return sets, labels


class Bootstrapped_Sets_Dataset(Dataset):
    def __init__(self):
        pass

    def generate(self, level, split, label, n, m, agg_data_dir, save_dir):
        save_dir = Path(save_dir)
        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        agg_dset = Aggregated_Raw_Dataset()
        agg_dset.load(level, split, label, agg_data_dir)

        sampled_sets, labels = bootstrap_sets(agg_dset.df, label, n, m)

        sampled_sets.to_pickle(feature_file_path)
        np.save(label_file_path, labels)
        
    def load(self, level, split, save_dir):
        save_dir = Path(save_dir)
        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        self.sets = pd.read_pickle(feature_file_path)
        self.labels = torch.from_numpy(np.load(label_file_path, allow_pickle=True))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.sets.loc[index]
        x = torch.from_numpy(x.to_numpy())
        y = self.labels[index]
        # y = torch.unsqueeze(y, 0)
        return x, y