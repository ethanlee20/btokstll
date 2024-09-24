
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def make_save_file_name(level, split):
    name = f"agg_sig_{level}_{split}.pkl"
    return name


def get_file_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def aggregate(level, split, columns, dir):
    dir = Path(dir)

    train_trials = (1, 31)
    eval_trials = (31, 41)
    
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
    
    data = [pd.read_pickle(path).loc[level][columns] for path in file_paths]
    labels = [get_file_info(path)["dc9"] for path in file_paths]
    labeled_data = [df.assign(dc9=dc9) for df, dc9 in zip(data, labels)]
    
    df = pd.concat(labeled_data)
    return df


class Aggregated_Signal_Dataset(Dataset):
    
    def __init__(self):
        pass
    
    def generate(self, level, split, features, signal_dir, save_dir):    
        save_dir = Path(save_dir)
        
        df = aggregate(level, split, features, signal_dir)

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