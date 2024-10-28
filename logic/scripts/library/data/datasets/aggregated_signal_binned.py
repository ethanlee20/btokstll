
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def make_feat_file_name(level, split):
    name = f"agg_sig_bin_feat_{level}_{split}.npy"
    return name


def make_label_file_name(level, split):
    name = f"agg_sig_bin_label_{level}_{split}.npy"
    return name


def make_bins_file_name(level, split):
    name = f"agg_sig_bin_bins_{level}_{split}.npy"
    return name


def get_file_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def to_bins(ar):
    bins, inv_ind = np.unique(ar, return_inverse=True)
    u_bin_ind = np.arange(len(bins))
    bin_ind = u_bin_ind[inv_ind]
    return bin_ind, bins


def aggregate(level, split, columns, dir, q2_veto=True, normalize=True, dtype="float64"):
    dir = Path(dir)

    train_trials = (1, 2)
    eval_trials = (2, 3)
    
    if split == "train": 
        trials = train_trials
    elif split == "eval": 
        trials = eval_trials
    else: 
        raise ValueError
    
    file_paths = []
    for path in list(dir.glob("*.pkl")):
        trial_num = get_file_info(path)["trial"]
        if trial_num in range(*trials):
            file_paths.append(path)
    
    l_df_feat = [pd.read_pickle(path).loc[level][columns] for path in file_paths]
    l_label = [get_file_info(path)["dc9"] for path in file_paths]
    df_labeled = pd.concat([df.assign(dc9=dc9) for df, dc9 in zip(l_df_feat, l_label)])
    df_labeled = df_labeled.astype(dtype)

    if q2_veto:
        df_labeled = df_labeled[(df_labeled["q_squared"]>1) & (df_labeled["q_squared"]<8)]

    ar_feat = df_labeled.drop(columns="dc9").to_numpy()
    if normalize:
        mean = np.mean(ar_feat, axis=0)
        stdev = np.std(ar_feat, axis=0)
        ar_feat = (ar_feat - mean) / stdev 
    
    ar_label = df_labeled["dc9"].to_numpy()

    ar_label_bin_ind, bins = to_bins(ar_label)

    return ar_feat, ar_label_bin_ind, bins


class Aggregated_Signal_Binned_Dataset(Dataset):
    
    def __init__(self):
        pass
    
    def generate(self, level, split, features, signal_dir, save_dir):    
        save_dir = Path(save_dir)
        
        ar_feat, ar_label, ar_bins = aggregate(level, split, features, signal_dir)
        
        feat_file_name = make_feat_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        bins_file_name = make_bins_file_name(level, split)
        feat_save_path = save_dir.joinpath(feat_file_name)
        label_save_path = save_dir.joinpath(label_file_name)
        bins_save_path = save_dir.joinpath(bins_file_name)

        np.save(feat_save_path, ar_feat)
        np.save(label_save_path, ar_label)
        np.save(bins_save_path, ar_bins)

    def load(self, level, split, save_dir):

        save_dir = Path(save_dir)
        feat_file_name = make_feat_file_name(level, split)
        label_file_name = make_label_file_name(level, split)
        bins_file_name = make_bins_file_name(level, split)
        feat_file_path = save_dir.joinpath(feat_file_name)
        label_file_path = save_dir.joinpath(label_file_name)
        bins_file_path = save_dir.joinpath(bins_file_name)
        ar_feat = np.load(feat_file_path, allow_pickle=True)
        ar_label = np.load(label_file_path, allow_pickle=True)
        ar_bins = np.load(bins_file_path, allow_pickle=True)
        
        self.feat = torch.from_numpy(ar_feat)
        self.labels = torch.from_numpy(ar_label)
        self.bins = torch.from_numpy(ar_bins)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.feat[index]
        y = self.labels[index]
        return x, y