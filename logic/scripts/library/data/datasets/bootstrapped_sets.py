
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .aggregated_signal import Aggregated_Signal_Dataset
from .background import Background_Dataset


def make_feature_file_name(level, split):
    assert level in {"gen", "det"}
    assert split in {"train", "eval", "lin_eval"}
    name = f"boot_sets_feat_{level}_{split}.pkl"
    return name


def make_label_file_name(level, split):
    assert level in {"gen", "det"}
    assert split in {"train", "eval", "lin_eval"}
    name = f"boot_sets_label_{level}_{split}.npy"
    return name


def bootstrap_labeled_sets(df, label, n, m):
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

    def generate(
            self, level, split, label, 
            n_signal, n_bkg, m, q2_veto, 
            agg_sig_data_dir, bkg_data_dir, save_dir
        ):
        save_dir = Path(save_dir)
        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        sig_dset = Aggregated_Signal_Dataset()
        if split == "train":
            sig_dset.load(level, "train", label, agg_sig_data_dir)
        elif split in {"eval", "lin_eval"}:
            sig_dset.load(level, "eval", label, agg_sig_data_dir)

        bkg_dset = Background_Dataset()
        if split == "train":
            bkg_dset.load("train", bkg_data_dir)
        elif split in {"eval", "lin_eval"}:
            bkg_dset.load("eval", bkg_data_dir)

        df_sig = sig_dset.df
        df_bkg_charge = bkg_dset.df_charge
        df_bkg_mix = bkg_dset.df_mix

        df_sig = df_sig.dropna(how="any")
        df_bkg_charge = df_bkg_charge.dropna(how="any")
        df_bkg_mix = df_bkg_mix.dropna(how="any")

        if q2_veto:
            df_sig = df_sig[df_sig["q_squared"]<8]
            df_bkg_charge = df_bkg_charge[df_bkg_charge["q_squared"]<8]
            df_bkg_mix = df_bkg_mix[df_bkg_mix["q_squared"]<8]

        df_signal_sets, labels = bootstrap_labeled_sets(
            df_sig, 
            label, n_signal, m
        )

        n_sets = m * len(labels)
        
        # Seeing a similar number of mixed bkg events to charged bkg events (after scaling for initial data size)
        bkg_charge_sets = [df_bkg_charge.sample(n=int(n_bkg/2), replace=True) for i in range(n_sets)]
        bkg_mix_sets = [df_bkg_mix.sample(n=int(n_bkg/2), replace=True) for i in range(n_sets)]
        
        df_bkg_charge_sets = pd.concat(bkg_charge_sets, keys=range(n_sets), names=["set", "event"])
        df_bkg_mix_sets = pd.concat(bkg_mix_sets, keys=range(n_sets), names=["set", "event"])

        df_sets  = pd.concat([df_signal_sets, df_bkg_charge_sets, df_bkg_mix_sets])
        # breakpoint()
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