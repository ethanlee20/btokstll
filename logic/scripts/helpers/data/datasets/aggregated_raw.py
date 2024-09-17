
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset


def make_save_file_name(split):
    name = f"aggregated_raw_{split}.pkl"
    return name


def get_raw_file_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def aggregate_raw(columns, trials, raw_data_dir):
    raw_data_dir = Path(raw_data_dir)

    filepaths = []
    for path in list(raw_data_dir.glob("*.pkl")):
        trial_num = get_raw_file_info(path)["trial"]
        if trial_num in range(*trials):
            filepaths.append(path)

    raw_data = [pd.read_pickle(path)[columns] for path in filepaths]
    dc9_values = [get_raw_file_info(path)["dc9"] for path in filepaths]
    raw_labeled_data = [df.assign(dc9=dc9) for df, dc9 in zip(raw_data, dc9_values)]
    aggregate_data = pd.concat(raw_labeled_data)
    return aggregate_data


class Aggregated_Raw_Dataset():
    
    def __init__(self):
        pass
    
    def generate(self, columns, split, raw_data_dir, save_dir):    
        save_dir = Path(save_dir)
        
        train_trial_range = (1, 31)
        eval_trial_range = (31, 41)
        
        if split == "train": 
            trials = train_trial_range
        elif split == "eval": 
            trials = eval_trial_range
        else: 
            raise ValueError
        
        df = aggregate_raw(columns, trials, raw_data_dir)

        save_file_name = make_save_file_name(split)
        save_path = save_dir.joinpath(save_file_name)

        pd.to_pickle(df, save_path)

    def load(self, split, save_dir):
        save_dir = Path(save_dir)
        save_file_name = make_save_file_name(split)
        save_path = save_dir.joinpath(save_file_name)
        df = pd.read_pickle(save_path)
        self.data = df
