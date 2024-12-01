
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .aggregated_signal_binned import Aggregated_Signal_Binned_Dataset
from .background import Background_Dataset


def make_feature_file_name(level, split):
    assert level in {"gen", "det"}
    assert split in {"train", "eval", "lin_eval"}
    name = f"boot_sets_bin_feat_{level}_{split}.pkl"
    return name


def make_label_file_name(level, split):
    assert level in {"gen", "det"}
    assert split in {"train", "eval", "lin_eval"}
    name = f"boot_sets_bin_label_{level}_{split}.npy"
    return name




class Bootstrapped_Sets_Binned_Dataset(Dataset):
    def __init__(self):
        pass

    def generate(
            self, level, split, 
            n_signal, n_bkg, m,
            agg_sig_data_dir, bkg_data_dir, save_dir
        ):

        sig_dset = Aggregated_Signal_Binned_Dataset()
        if split == "train":
            sig_dset.load(level, "train", agg_sig_data_dir)
        elif split in {"eval", "lin_eval"}:
            sig_dset.load(level, "eval", agg_sig_data_dir)

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

        df_signal_sets, labels = bootstrap_labeled_sets(
            df_sig, 
            "dc9_bin", n_signal, m
        )

        num_sets = len(labels)
        
        # Seeing a similar number of mixed bkg events to charged bkg events (after scaling for initial data size)
        bkg_charge_sets = [df_bkg_charge.sample(n=int(n_bkg/2), replace=True) for _ in range(num_sets)]
        bkg_mix_sets = [df_bkg_mix.sample(n=int(n_bkg/2), replace=True) for _ in range(num_sets)]
        
        df_bkg_charge_sets = pd.concat(bkg_charge_sets, keys=range(num_sets), names=["set", "event"])
        df_bkg_mix_sets = pd.concat(bkg_mix_sets, keys=range(num_sets), names=["set", "event"])

        df_sets = pd.concat([df_signal_sets, df_bkg_charge_sets, df_bkg_mix_sets])

        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        save_dir = Path(save_dir)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)
        
        df_sets.to_pickle(feature_file_path)
        np.save(label_file_path, labels)
        
    def load(self, level, split, save_dir):
        feature_file_name = make_feature_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        save_dir = Path(save_dir)
        feature_file_path = save_dir.joinpath(feature_file_name)
        label_file_path = save_dir.joinpath(label_file_name)

        df_sets = pd.read_pickle(feature_file_path)
        labels = np.load(label_file_path, allow_pickle=True)

        self.sets = torch.from_numpy(np.concatenate([np.expand_dims(s, axis=0) for _, s in df_sets.groupby(level="set")], axis=0))
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.sets[index]
        y = self.labels[index]
        return x, y