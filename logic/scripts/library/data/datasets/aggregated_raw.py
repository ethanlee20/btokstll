
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


def aggregate_raw(level, trials:tuple, columns, raw_data_dir):
    raw_data_dir = Path(raw_data_dir)

    filepaths = []
    for path in list(raw_data_dir.glob("*.pkl")):
        trial_num = get_raw_file_info(path)["trial"]
        if trial_num in range(*trials):
            filepaths.append(path)

    raw_data = [pd.read_pickle(path).loc[level][columns] for path in filepaths]
    dc9_values = [get_raw_file_info(path)["dc9"] for path in filepaths]
    raw_labeled_data = [df.assign(dc9=dc9) for df, dc9 in zip(raw_data, dc9_values)]
    aggregate_data = pd.concat(raw_labeled_data)
    return aggregate_data


class Aggregated_Raw_Dataset(Dataset):
    
    def __init__(self):
        pass
    
    def generate(self, level, split, columns, raw_data_dir, save_dir, sample=None):    
        save_dir = Path(save_dir)
        
        train_trial_range = (1, 31)
        eval_trial_range = (31, 41)
        
        if split == "train": 
            trials = train_trial_range
        elif split == "eval": 
            trials = eval_trial_range
        else: 
            raise ValueError
        
        df = aggregate_raw(level, trials, columns, raw_data_dir)

        if sample is not None:
            df = df.sample(n=sample, replace=True)

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