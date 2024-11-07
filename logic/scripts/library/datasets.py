
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_raw_datafile_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def aggregate_raw_signal(level, raw_trials:range, columns:list[str], raw_signal_dir_path, dtype="float64"):
    raw_signal_dir_path = Path(raw_signal_dir_path)
    
    raw_datafile_paths = []
    for raw_datafile_path in list(raw_signal_dir_path.glob("*.pkl")):
        raw_datafile_trial_number = get_raw_datafile_info(raw_datafile_path)["trial"]
        if raw_datafile_trial_number in raw_trials:
            raw_datafile_paths.append(raw_datafile_path)
    
    loaded_dataframes = [pd.read_pickle(path).loc[level][columns] for path in raw_datafile_paths]
    loaded_dataframe_dc9_values = [get_raw_datafile_info(path)["dc9"] for path in raw_datafile_paths]
    labeled_dataframe = pd.concat([df.assign(dc9=dc9) for df, dc9 in zip(loaded_dataframes, loaded_dataframe_dc9_values)])
    labeled_dataframe = labeled_dataframe.astype(dtype)
    
    return labeled_dataframe


def apply_q_squared_veto(df: pd.DataFrame):
    lower_bound = 1
    upper_bound = 8
    df_vetoed = df[(df["q_squared"]>lower_bound) & (df["q_squared"]<upper_bound)]
    return df_vetoed


def apply_std_scale(df: pd.DataFrame):
    df_scaled = (df - df.mean()) / df.std()
    return df_scaled    


def to_bins(ar):
    ar = np.array(ar)
    bin_values, inverse_indices = np.unique(ar, return_inverse=True)
    bin_indices = np.arange(len(bin_values))
    bins = bin_indices[inverse_indices]
    return bins, bin_values


def balance_classes(df: pd.DataFrame, label_column_name: str):
    group_by_label = df.groupby(label_column_name)
    num_events = [len(df_label) for _, df_label in group_by_label]
    min_num_events = min(num_events)
    balanced_dfs = [df_label[:min_num_events] for _, df_label in group_by_label]
    balanced_df = pd.concat(balanced_dfs)
    return balanced_df


class Aggregated_Signal_Binned_Dataset(Dataset):
    
    def __init__(self, level, split, raw_trials, save_dir):
        self.level = level
        self.split = split
        self.raw_trials = raw_trials
        self.save_dir = Path(save_dir)
        
        self.features = ["q_squared", "costheta_mu", "costheta_K", "chi"]
        self.dc9_column_name = "dc9" # defined elsewhere as well: can fix this
        self.dc9_bin_column_name = "dc9_bin_index"

        dataframe_file_name = self.make_dataframe_save_file_name()
        bin_values_file_name = self.make_bin_values_save_file_name()
        self.dataframe_file_save_path = self.save_dir.joinpath(dataframe_file_name)
        self.bin_values_file_save_path = self.save_dir.joinpath(bin_values_file_name)

    def make_dataframe_save_file_name(self):
        name = f"agg_sig_bin_df_{self.level}_{self.split}.pkl"
        return name

    def make_bin_values_save_file_name(self):
        name = f"agg_sig_bin_values_{self.level}_{self.split}.npy"
        return name

    def generate(self, raw_signal_dir, q_squared_veto=True, std_scale=True, balanced_classes=True):    
        df_agg = aggregate_raw_signal(self.level, self.raw_trials, self.features, raw_signal_dir)

        bins, bin_values = to_bins(df_agg[self.dc9_column_name])
        df_agg[self.dc9_bin_column_name] = bins
        df_agg = df_agg.drop(columns=self.dc9_column_name)
        
        if q_squared_veto:
            df_agg = apply_q_squared_veto(df_agg)

        if balanced_classes:
            df_agg = balance_classes(df_agg, self.dc9_bin_column_name)

        if std_scale:
            df_agg[self.features] = apply_std_scale(df_agg[self.features])

        df_agg.to_pickle(self.dataframe_file_save_path)
        np.save(self.bin_values_file_save_path, bin_values)

    def load(self):
        self.df = pd.read_pickle(self.dataframe_file_save_path)
        self.bin_values = np.load(self.bin_values_file_save_path, allow_pickle=True)
        
        self.feat = torch.from_numpy(self.df[self.features].to_numpy())
        self.labels = torch.from_numpy(self.df[self.dc9_bin_column_name].to_numpy())

    def to(self, device):
        self.feat = self.feat.to(device)
        self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.feat[index]
        y = self.labels[index]
        return x, y