
"""Dataset definitions and utilities."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from library.util import bootstrap_labeled_sets


def get_raw_datafile_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def aggregate_raw_signal(level, raw_trials:range, columns:list[str], raw_signal_dir_path, dtype="float64"):
    """
    Aggregate data from specified raw signal files.
    
    Returns
    ------
    A dataframe with specified feature columns and a label column named "dc9".
    """
    raw_signal_dir_path = Path(raw_signal_dir_path)
    
    raw_datafile_paths = []
    for raw_datafile_path in list(raw_signal_dir_path.glob("*.pkl")):
        raw_datafile_trial_number = get_raw_datafile_info(raw_datafile_path)["trial"]
        if raw_datafile_trial_number in raw_trials:
            raw_datafile_paths.append(raw_datafile_path)
    
    num_files_to_load = len(raw_datafile_paths)
    loaded_dataframes = []
    for file_number, path in enumerate(raw_datafile_paths, start=1):
        loaded_dataframe = pd.read_pickle(path).loc[level][columns]
        print(f"opened raw file: [{file_number}/{num_files_to_load}] {path}")
        loaded_dataframes.append(loaded_dataframe)
    loaded_dataframe_dc9_values = [get_raw_datafile_info(path)["dc9"] for path in raw_datafile_paths]
    labeled_dataframe = pd.concat([df.assign(dc9=dc9) for df, dc9 in zip(loaded_dataframes, loaded_dataframe_dc9_values)])
    labeled_dataframe = labeled_dataframe.astype(dtype)
    return labeled_dataframe


def make_image(df, n_bins):
    """
    Make an image of a dataset. (Like Shawn.)

    Parameters
    ----------
    df : Dataframe of dataset
    n_bins : The number of bins of each angular variable.

    Returns
    -------
    pytorch Tensor image. 
    """
    df = df.copy()
    bins = {
        "chi": np.linspace(start=0, stop=2*np.pi, num=n_bins+1),
        "costheta_mu": np.linspace(start=-1, stop=1, num=n_bins+1),
        "costheta_K": np.linspace(start=-1, stop=1, num=n_bins+1),
    }
    var_col_names = [*bins.keys()]
    bin_col_names = [var_col_name + "_bin" for var_col_name in var_col_names]
    for var_col_name, bin_col_name in zip(var_col_names, bin_col_names):
        df[bin_col_name] = pd.cut(df[var_col_name], bins=bins[var_col_name], include_lowest=True)
    df_image = df.groupby(bin_col_names, observed=False).mean()["q_squared"]
    np_image = df_image.to_numpy().reshape((n_bins,)*3) # dimensions of (chi, costheta_mu, costheta_K)
    np_image = np.expand_dims(np_image, axis=0)
    np_image = np.nan_to_num(np_image)
    torch_image = torch.from_numpy(np_image)
    return torch_image


def apply_q_squared_veto(df: pd.DataFrame):
    lower_bound = 1
    upper_bound = 8
    df_vetoed = df[(df["q_squared"]>lower_bound) & (df["q_squared"]<upper_bound)]
    return df_vetoed


def apply_std_scale_dataframe(df: pd.DataFrame):
    df_scaled = (df - df.mean()) / df.std()
    return df_scaled


def apply_std_scale_numpy_sets(a: np.ndarray):
    """Standard scale each set in an array of sets."""
    mu = np.expand_dims(np.mean(a, axis=(-1,-2)), axis=(-1,-2))
    sigma = np.expand_dims(np.std(a, axis=(-1, -2)), axis=(-1, -2))
    a_scaled = (a - mu) / sigma    
    return a_scaled


def to_bins(ar):
    ar = np.array(ar)
    bin_values, inverse_indices = np.unique(ar, return_inverse=True)
    bin_indices = np.arange(len(bin_values))
    bins = bin_indices[inverse_indices]
    return bins, bin_values


def balance_classes(df: pd.DataFrame, label_column_name: str):
    """Reduce the number of events per label to the minimum over the labels."""
    group_by_label = df.groupby(label_column_name)
    num_events = [len(df_label) for _, df_label in group_by_label]
    min_num_events = min(num_events)
    balanced_dfs = [df_label[:min_num_events] for _, df_label in group_by_label]
    balanced_df = pd.concat(balanced_dfs)
    return balanced_df


means = {         # with q squared veto, from first 5 trials # 
    "gen": {
        "q_squared": 4.75234051e+00,
        "costheta_mu": 6.51485574e-02,
        "costheta_K": 6.42255401e-05,
        "chi": 3.14130932e+00, 
    }, 
    "det": {
        "q_squared": None,  # not calculated yet
        "costheta_mu": None,
        "costheta_K": None,
        "chi": None, 
    }
}

stds = {         # no q squared veto, from first 5 trials # 
    "gen": {
        "q_squared": 2.05356902,
        "costheta_mu": 0.50583777,
        "costheta_K": 0.69434327,
        "chi": 1.81157865, 
    }, 
    "det": {
        "q_squared": None,  # not calculated yet
        "costheta_mu": None,
        "costheta_K": None,
        "chi": None, 
    }
}


class Aggregated_Signal_Binned_Dataset(Dataset):
    
    def __init__(self, level, split, save_dir, feature_names=["q_squared", "costheta_mu", "costheta_K", "chi"]):
        self.level = level
        self.split = split
        self.save_dir = Path(save_dir)
        
        self.feature_names = feature_names
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

    def generate(self, raw_trials, raw_signal_dir, q_squared_veto=True, std_scale=True, balanced_classes=True, shuffle=True):    
        df_agg = aggregate_raw_signal(self.level, raw_trials, self.feature_names, raw_signal_dir)

        bins, bin_values = to_bins(df_agg[self.dc9_column_name])
        df_agg[self.dc9_bin_column_name] = bins
        df_agg = df_agg.drop(columns=self.dc9_column_name)
        
        if q_squared_veto:
            df_agg = apply_q_squared_veto(df_agg)

        if std_scale:
            for column_name in self.feature_names:
                df_agg[column_name] = (df_agg[column_name] - means[self.level][column_name]) / stds[self.level][column_name]

        if balanced_classes:
            df_agg = balance_classes(df_agg, self.dc9_bin_column_name)

        if shuffle:
            df_agg = df_agg.sample(frac=1)

        df_agg.to_pickle(self.dataframe_file_save_path)
        np.save(self.bin_values_file_save_path, bin_values)

    def load(self, device="cpu"):
        self.df = pd.read_pickle(self.dataframe_file_save_path)
        self.bin_values = np.load(self.bin_values_file_save_path, allow_pickle=True)
        
        self.features = torch.from_numpy(self.df[self.feature_names].to_numpy())
        self.labels = torch.from_numpy(self.df[self.dc9_bin_column_name].to_numpy())

        if device != "cpu":
            self.to(device)

        print(f"Loaded data with features size: {self.features.shape} and labels size: {self.labels.shape}.")

    def to(self, device):
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    

class Bootstrapped_Signal_Binned_Dataset(Dataset):

    def __init__(self, level, split, save_dir):
        self.level = level
        self.split = split
        self.save_dir = Path(save_dir)
        
        self.feature_names = ["q_squared", "costheta_mu", "costheta_K", "chi"]
        self.dc9_column_name = "dc9" # defined elsewhere as well: can fix this
        self.dc9_bin_column_name = "dc9_bin_index"

        features_file_name = self.make_features_save_file_name()
        labels_file_name = self.make_labels_save_file_name()
        bin_values_file_name = self.make_bin_values_save_file_name()
        self.features_save_path = self.save_dir.joinpath(features_file_name)
        self.labels_save_path = self.save_dir.joinpath(labels_file_name)
        self.bin_values_save_path = self.save_dir.joinpath(bin_values_file_name)

    def make_features_save_file_name(self):
        name = f"boot_sig_bin_feat_{self.level}_{self.split}.pt"
        return name

    def make_labels_save_file_name(self):
        name = f"boot_sig_bin_lab_{self.level}_{self.split}.pt"
        return name

    def make_bin_values_save_file_name(self):
        name = f"boot_sig_bin_values_{self.level}_{self.split}.npy"
        return name

    def generate(
        self, raw_trials, raw_signal_dir, 
        num_events_per_set, num_sets_per_label, 
        q_squared_veto=True, std_scale=True, balanced_classes=True,
    ):
        df_agg = aggregate_raw_signal(self.level, raw_trials, self.feature_names, raw_signal_dir)
        
        bins, bin_values = to_bins(df_agg[self.dc9_column_name])
        np.save(self.bin_values_save_path, bin_values)
        
        df_agg[self.dc9_bin_column_name] = bins
        df_agg = df_agg.drop(columns=self.dc9_column_name)
        
        if q_squared_veto:
            df_agg = apply_q_squared_veto(df_agg)

        if std_scale:
            for column_name in self.feature_names:
                df_agg[column_name] = (df_agg[column_name] - means[self.level][column_name]) / stds[self.level][column_name]

        if balanced_classes:
            df_agg = balance_classes(df_agg, self.dc9_bin_column_name)

        df_sets = bootstrap_labeled_sets(
            df_agg,
            self.dc9_bin_column_name,
            num_events_per_set, num_sets_per_label
        )

        set_labels = df_sets.index.get_level_values("label")[::num_events_per_set]
        labels = torch.from_numpy(set_labels.to_numpy())
        torch.save(labels, self.labels_save_path)

        num_sets = len(labels)
        set_indices = range(num_sets)
        list_of_sets = [df_sets.xs(i, level="set") for i in set_indices]
        expanded_list_of_sets = [np.expand_dims(s, axis=0) for s in list_of_sets]
        ndarray_of_sets = np.concatenate(expanded_list_of_sets, axis=0)
        features = torch.from_numpy(ndarray_of_sets)
        torch.save(features, self.features_save_path)

    def load(self, device="cpu"): 
        self.bin_values = np.load(self.bin_values_save_path, allow_pickle=True)
        self.features = torch.load(self.features_save_path, weights_only=True)
        self.labels = torch.load(self.labels_save_path, weights_only=True)
        print(f"Loaded data with features size: {self.features.shape} and labels size: {self.labels.shape}.")

        if device != "cpu":
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    

class Signal_Unbinned_Dataset(Dataset):
    """Torch dataset of unbinned signal events."""

    def __init__(self, level, split, save_dir, feature_names=["q_squared", "costheta_mu", "costheta_K", "chi"]):
        """Setup paths and names."""
        self.level = level
        self.split = split
        self.save_dir = Path(save_dir)

        self.feature_names = feature_names
        self.label_name = "dc9" # defined elsewhere as well

        features_file_name = self.make_features_file_name()
        labels_file_name = self.make_labels_file_name()
        self.features_file_path = self.save_dir.joinpath(features_file_name)
        self.labels_file_path = self.save_dir.joinpath(labels_file_name)

    def make_features_file_name(self):
        return f"signal_unbinned_ebe_feat_{self.level}_{self.split}.pt"
    
    def make_labels_file_name(self):
        return f"signal_unbinned_ebe_labels_{self.level}_{self.split}.pt"

    def generate(self, raw_trials, raw_signal_dir,  
        q_squared_veto=True, std_scale=True, balanced_classes=True,
        dtype="float32"
    ):
        """Generate and save dataset state."""

        df_agg = aggregate_raw_signal(self.level, raw_trials, self.feature_names, raw_signal_dir, dtype=dtype)
        
        if q_squared_veto:
            df_agg = apply_q_squared_veto(df_agg)

        if std_scale:
            for column_name in self.feature_names:
                df_agg[column_name] = (df_agg[column_name] - means[self.level][column_name]) / stds[self.level][column_name]

        if balanced_classes:
            df_agg = balance_classes(df_agg, self.label_name)

        labels = df_agg[[self.label_name]]
        features = df_agg[self.feature_names]

        labels = torch.from_numpy(labels.to_numpy())
        features = torch.from_numpy(features.to_numpy())

        torch.save(labels, self.labels_file_path)
        torch.save(features, self.features_file_path)

    def load(self, device="cpu"):
        """Load dataset state."""

        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        if device != "cpu":
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)
   
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y


class Bootstrapped_Signal_Unbinned_Dataset(Dataset):
    """Torch dataset of bootstrapped sets of unbinned signal events."""

    def __init__(self, level, split, save_dir, feature_names=["q_squared", "costheta_mu", "costheta_K", "chi"]):
        """
        Setup paths and things.

        level : "gen" or "det"
        split : "train" or "val" (or custom)
        save_dir : path to directory where the dataset state should be saved and loaded from.
        """
        self.level = level
        self.split = split
        self.save_dir = Path(save_dir)
        
        self.feature_names = feature_names
        self.label_name = "dc9" # defined elsewhere as well: can fix this

        features_file_name = self._make_features_file_name()
        labels_file_name = self._make_labels_file_name()
        self.features_file_path = self.save_dir.joinpath(features_file_name)
        self.labels_file_path = self.save_dir.joinpath(labels_file_name)

    def _make_features_file_name(self):
        name = f"signal_unbinned_sets_feat_{self.level}_{self.split}.pt"
        return name
    
    def _make_labels_file_name(self):
        name = f"signal_unbinned_sets_labels_{self.level}_{self.split}.pt"
        return name

    def generate(
        self, raw_trials, raw_signal_dir, 
        num_events_per_set, num_sets_per_label, 
        q_squared_veto=True, std_scale=True, balanced_classes=True,
        dtype="float32"
    ):
        """
        Generate and save dataset state.
        """
        df_agg = aggregate_raw_signal(self.level, raw_trials, self.feature_names, raw_signal_dir, dtype=dtype)
       
        if q_squared_veto:
            df_agg = apply_q_squared_veto(df_agg)

        if std_scale:
            for column_name in self.feature_names:
                df_agg[column_name] = (df_agg[column_name] - means[self.level][column_name]) / stds[self.level][column_name]

        if balanced_classes:
            df_agg = balance_classes(df_agg, self.label_name)

        df_sets = bootstrap_labeled_sets(
            df_agg,
            self.label_name,
            num_events_per_set, num_sets_per_label
        )

        # labels
        event_labels = df_sets.index.get_level_values("label").astype(dtype)
        set_labels = event_labels[::num_events_per_set]
        set_labels = torch.from_numpy(set_labels.to_numpy())
        set_labels = set_labels.unsqueeze(1)
        torch.save(set_labels, self.labels_file_path)

        # features
        num_sets = len(set_labels)
        set_indices = range(num_sets)
        list_of_sets = [df_sets.xs(i, level="set") for i in set_indices]
        expanded_list_of_sets = [np.expand_dims(s, axis=0) for s in list_of_sets]
        ndarray_of_sets = np.concatenate(expanded_list_of_sets, axis=0)

        set_features = torch.from_numpy(ndarray_of_sets)
        torch.save(set_features, self.features_file_path)

    def load(self, device="cpu"): 
        """
        Load saved dataset state to specified device. 
        """
        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        if device != "cpu":
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    

class Signal_Images_Dataset(Dataset):
    """Bootstrapped images (like Shawn)."""
    def __init__(self, level, split, save_dir, feature_names=["q_squared", "costheta_mu", "costheta_K", "chi"]):
        self.level = level
        self.split = split
        self.save_dir = Path(save_dir)
        self.feature_names = feature_names
        self.features_save_path = self.save_dir.joinpath(self.make_features_file_name())
        self.labels_save_path = self.save_dir.joinpath(self.make_labels_file_name())

    def make_features_file_name(self):
        name = f"img_sig_feat_{self.level}_{self.split}.pt"
        return name
    
    def make_labels_file_name(self):
        name = f"img_sig_label_{self.level}_{self.split}.pt"
        return name
    
    def generate(
            self, raw_trials:range, raw_signal_dir, 
            num_events_per_set, num_sets_per_label, n_bins,
            q_squared_veto=True, std_scale=True, balanced_classes=True,
        ):
        df_agg = aggregate_raw_signal(self.level, raw_trials, self.feature_names, raw_signal_dir)

        if q_squared_veto:
            df_agg = apply_q_squared_veto(df_agg)

        if std_scale:
            df_agg["q_squared"] = (df_agg["q_squared"] - means[self.level]["q_squared"]) / stds[self.level]["q_squared"]

        if balanced_classes:
            df_agg = balance_classes(df_agg, label_column_name="dc9")

        df_sets = bootstrap_labeled_sets(df_agg, label="dc9", n=num_events_per_set, m=num_sets_per_label)
        set_labels = df_sets.index.get_level_values("label")[::num_events_per_set]
        set_labels = torch.from_numpy(set_labels.to_numpy())

        num_sets = len(set_labels)
        set_indices = range(num_sets)
        list_of_set_dfs = [df_sets.xs(i, level="set") for i in set_indices]
        list_of_images = [make_image(df, n_bins=n_bins) for df in list_of_set_dfs]
        list_of_expanded_images = [torch.unsqueeze(image, axis=0) for image in list_of_images]
        images_tensor = torch.concatenate(list_of_expanded_images, axis=0)
        
        torch.save(images_tensor, self.features_save_path)
        torch.save(set_labels, self.labels_save_path)

    def load(self, device="cpu"): 
        self.features = torch.load(self.features_save_path, weights_only=True)
        self.labels = torch.load(self.labels_save_path, weights_only=True)
        print(f"Loaded data with features size: {self.features.shape} and labels size: {self.labels.shape}.")

        if device != "cpu":
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y




    

        


