
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

    def generate(self, level, split, label, n_signal, n_bkg, m, allow_mis:bool, agg_data_dir, save_dir):
        save_dir = Path(save_dir)
        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        agg_dset = Aggregated_Raw_Dataset()
        agg_dset.load(level, split, label, agg_data_dir)

        df_agg = agg_dset.df

        signal_source_id = 0
        misrecon_source_id = 1
        charge_bkg_source_id = 2
        mix_bkg_source_id = 3

        signal = (df_agg["source_id"] == signal_source_id)
        misrecon = (df_agg["source_id"] == misrecon_source_id)
        charge_bkg = (df_agg["source_id"] == charge_bkg_source_id)
        mix_bkg = (df_agg["source_id"] == mix_bkg_source_id)

        if allow_mis:
            signal_sets, labels = bootstrap_sets(
                df_agg[signal|misrecon], 
                label, n_signal, m
            )
        elif not allow_mis:
            signal_sets, labels = bootstrap_sets(
                df_agg[signal],
                label, n_signal, m
            )

        # Seeing a similar number of mixed bkg events to charged bkg events (after scaling for initial data size)
        charge_bkg_sets = [df_agg[charge_bkg].sample(n=int(n_bkg/2), replace=True) for i in range(m)]
        charge_bkg_sets = pd.concat(charge_bkg_sets, keys=range(m), names=["set", "event"])
        mix_bkg_sets = [df_agg[mix_bkg].sample(n=int(n_bkg/2), replace=True) for i in range(m)]
        mix_bkg_sets = pd.concat(mix_bkg_sets, keys=range(m), names=["set", "event"])

        df_sets  = pd.concat([signal_sets, charge_bkg_sets, mix_bkg_sets])

        df_sets.to_pickle(feature_file_path)
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