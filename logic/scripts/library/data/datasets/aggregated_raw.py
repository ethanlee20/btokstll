
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def make_save_file_name(level, split):
    name = f"aggregated_raw_{level}_{split}.pkl"
    return name


def get_raw_file_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def aggregate_raw(level, trials:tuple, columns, signal_dir, mix_bkg_file_path, charge_bkg_file_path):
    signal_dir = Path(signal_dir)
    mix_bkg_file_path = Path(mix_bkg_file_path)
    charge_bkg_file_path = Path(charge_bkg_file_path)

    signal_file_paths = []
    for path in list(signal_dir.glob("*.pkl")):
        trial_num = get_raw_file_info(path)["trial"]
        if trial_num in range(*trials):
            signal_file_paths.append(path)
    signal_data = [pd.read_pickle(path).loc[level][columns] for path in signal_file_paths]
    dc9_values = [get_raw_file_info(path)["dc9"] for path in signal_file_paths]
    labeled_data = [df.assign(dc9=dc9) for df, dc9 in zip(signal_data, dc9_values)]
    df_signal = pd.concat(labeled_data)

    df_mix_bkg = pd.read_pickle(mix_bkg_file_path)
    df_charge_bkg = pd.read_pickle(charge_bkg_file_path)

    df_signal[df_signal["isSignal"]==1]["source_id"] = 0    # signal      : id 0
    df_signal[df_signal["isSignal"]!=1]["source_id"] = 1    # mis. recon. : id 1
    df_charge_bkg["source_id"] = 2                          # charged bkg : id 2
    df_mix_bkg["source_id"] = 3                             # mixed bkg   : id 3

    df_agg = pd.concat([df_signal, df_charge_bkg, df_mix_bkg])
    return df_agg


class Aggregated_Raw_Dataset(Dataset):
    
    def __init__(self):
        pass
    
    def generate(self, level, split, columns, signal_dir, mix_bkg_file_path, charge_bkg_file_path, save_dir):    
        save_dir = Path(save_dir)
        
        train_trial_range = (1, 31)
        eval_trial_range = (31, 41)
        
        if split == "train": 
            trials = train_trial_range
        elif split == "eval": 
            trials = eval_trial_range
        else: 
            raise ValueError
        
        df = aggregate_raw(level, trials, columns, signal_dir, mix_bkg_file_path, charge_bkg_file_path)

        save_file_name = make_save_file_name(level, split)
        save_path = save_dir.joinpath(save_file_name)

        pd.to_pickle(df, save_path)

    def load(self, level, split, label, save_dir):
        
        self.label = label

        save_dir = Path(save_dir)
        save_file_name = make_save_file_name(level, split)
        save_path = save_dir.joinpath(save_file_name)
        df = pd.read_pickle(save_path)
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        event = self.df.iloc[index]
        x = event.drop(self.label).values
        y = np.array(event[self.label])

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        y = torch.unsqueeze(y, 0)

        return x, y