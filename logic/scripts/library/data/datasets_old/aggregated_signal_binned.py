
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def make_df_file_name(level, split):
    name = f"agg_sig_bin_df_{level}_{split}.pkl"
    return name


def make_bin_map_file_name(level, split):
    name = f"agg_sig_bin_bin_map_{level}_{split}.npy"
    return name


def get_file_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def to_bins(ar):
    ar = np.array(ar)
    bin_map, inverse_indices = np.unique(ar, return_inverse=True)
    bin_indices = np.arange(len(bin_map))
    bins = bin_indices[inverse_indices]
    return bins, bin_map


def aggregate(level, trials, columns, raw_data_directory, q2_veto=True, normalize=True, dtype="float64"):
    raw_data_directory = Path(raw_data_directory)

    trials_to_load = trials
    
    raw_data_file_paths = []
    for raw_data_file_path in list(raw_data_directory.glob("*.pkl")):
        data_file_trial_number = get_file_info(raw_data_file_path)["trial"]
        if data_file_trial_number in trials_to_load:
            raw_data_file_paths.append(raw_data_file_path)
    
    list_of_feature_dataframes = [pd.read_pickle(path).loc[level][columns] for path in raw_data_file_paths]
    list_of_labels = [get_file_info(path)["dc9"] for path in raw_data_file_paths]
    dataframe_features_and_labels = pd.concat([df.assign(dc9=dc9) for df, dc9 in zip(list_of_feature_dataframes, list_of_labels)])
    dataframe_features_and_labels = dataframe_features_and_labels.astype(dtype)

    if q2_veto:
        apply_q2_cut = lambda x: x[(x["q_squared"]>1) & (x["q_squared"]<8)]
        dataframe_features_and_labels = apply_q2_cut(dataframe_features_and_labels)
    
    if normalize:
        apply_std_scale = lambda x: (x - x.mean()) / x.std()
        features = ["q_squared", "costheta_mu", "costheta_K", "chi"]
        dataframe_features_and_labels[features] = apply_std_scale(dataframe_features_and_labels[features])
    
    binned_labels, bin_map = to_bins(dataframe_features_and_labels["dc9"])
    dataframe_features_and_labels["dc9_bin"] = binned_labels

    dataframe_features_and_labels = dataframe_features_and_labels.drop(columns="dc9")

    return bin_map, dataframe_features_and_labels


class Aggregated_Signal_Binned_Dataset(Dataset):
    
    def __init__(self):
        pass
    
    def generate(self, level, split, features, signal_dir, save_dir):    

        trials_to_load = {"train":range(1,31), "eval": range(31,41)}
        
        bin_map, df = aggregate(level, trials_to_load[split], features, signal_dir)
        
        df_file_name = make_df_file_name(level, split)
        bins_file_name = make_bin_map_file_name(level, split)
        save_dir = Path(save_dir)
        df_save_path = save_dir.joinpath(df_file_name)
        bins_save_path = save_dir.joinpath(bins_file_name)

        df.to_pickle(df_save_path)
        np.save(bins_save_path, bin_map)

    def load(self, level, split, save_dir):

        save_dir = Path(save_dir)
        df_file_name = make_df_file_name(level, split)
        bins_file_name = make_bin_map_file_name(level, split)
        df_file_path = save_dir.joinpath(df_file_name)
        bins_file_path = save_dir.joinpath(bins_file_name)

        self.df = pd.read_pickle(df_file_path)
        self.bins = np.load(bins_file_path, allow_pickle=True)
        
        self.feat = torch.from_numpy(self.df[["q_squared", "costheta_mu", "costheta_K", "chi"]].to_numpy())
        self.labels = torch.from_numpy(self.df["dc9_bin"].to_numpy())

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.feat[index]
        y = self.labels[index]
        return x, y