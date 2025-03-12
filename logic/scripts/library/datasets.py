
"""Dataset definitions and utilities."""

import pathlib

import numpy
from scipy.stats import binned_statistic_dd
import pandas
import torch

from library.util import aggregate_raw_signal, bootstrap_labeled_sets


def load_aggregated_raw_signal(level:str, split:str, dir_path):
    if split == "train":
        trial_interval = (1,20)
    elif split == "eval":
        trial_interval = (21, 40)
    else:
        raise ValueError
    
    dir_path = pathlib.Path(dir_path)
    file_name = f"agg_sig_{trial_interval[0]}_to_{trial_interval[1]}_{level}.pkl"
    file_path = dir_path.joinpath(file_name)
    df_agg = pandas.read_pickle(file_path)
    print(f"Loaded aggregated raw signal file: {file_path}")
    return df_agg


## Preprocessing ##

def apply_q_squared_veto(df: pandas.DataFrame):
    lower_bound = 1
    upper_bound = 8
    df_vetoed = df[(df["q_squared"]>lower_bound) & (df["q_squared"]<upper_bound)]
    return df_vetoed


def balance_classes(df: pandas.DataFrame, label_column_name: str):
    """Reduce the number of events per label to the minimum over the labels."""
    group_by_label = df.groupby(label_column_name)
    num_events = [len(df_label) for _, df_label in group_by_label]
    min_num_events = min(num_events)
    balanced_dfs = [df_label[:min_num_events] for _, df_label in group_by_label]
    balanced_df = pandas.concat(balanced_dfs)
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


def to_bins(ar):
    ar = numpy.array(ar)
    bin_values, inverse_indices = numpy.unique(ar, return_inverse=True)
    bin_indices = numpy.arange(len(bin_values))
    bins = bin_indices[inverse_indices]
    return bins, bin_values


def make_image(set_features, n_bins):
    """
    Make an image of a dataset. Like Shawn.

    Order of columns must be:             
        "q_squared", 
        "costheta_mu", 
        "costheta_K", 
        "chi"

    Parameters
    ----------
    set_features : array
        Array of features for each example 
        of a set
    n_bins : int
        Number of bins per dimension

    Returns
    -------
    image : torch array
        3 dimensional array. Average q^2
        calculated per 3D angular bin.
        Shape of (n_bins, n_bins, n_bins).
    """
    angular_features = set_features[:,1:]
    q_squared_features = set_features[:,0]
    stats = binned_statistic_dd(
        sample=angular_features, 
        values=q_squared_features, 
        bins=n_bins,
        range=[(-1, 1), (-1, 1), (0, 2*numpy.pi)]
    )
    image = torch.from_numpy(stats.statistic)
    image = torch.nan_to_num(image)
    image = image.unsqueeze(0)
    return image




## Datasets ##

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, name, level, split, save_dir, extra_description=None):
        self.name = name 
        self.extra_description = extra_description
        self.level = level
        self.split = split
        self.save_dir = pathlib.Path(save_dir)

        self.features_file_path = self.make_file_path("features")
        self.labels_file_path = self.make_file_path("labels")

        self.feature_names = [
            "q_squared", 
            "costheta_mu", 
            "costheta_K", 
            "chi"
        ]
        self.label_name = "dc9"
        self.binned_label_name = "dc9_bin_index"

    def load(self):
        """
        Overwrite this with a function that loads some files
        (at least set self.features and self.labels).
        """

    def unload(self):
        """
        Overwrite this with a function that unloads the
        loaded data from memory (self.features and self.labels).
        """

    def generate(self):
        """
        Overwrite this with a function that saves some files
        (at least features and labels).
        """
        pass
    
    def make_file_path(self, kind):
        file_name = (
            f"{self.name}_{self.extra_description}_{self.level}_{self.split}_{kind}.pt"
            if self.extra_description
            else f"{self.name}_{self.level}_{self.split}_{kind}.pt"
        )
        file_path = self.save_dir.joinpath(file_name)
        return file_path

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y


class Binned_Signal_Dataset(Custom_Dataset):
    
    def __init__(
            self, 
            level, 
            split, 
            save_dir, 
            q_squared_veto=True,
            std_scale=True,
            balanced_classes=True,
            shuffle=True,
            extra_description=None,
            regenerate=False,
    ):
        super().__init__(
            "binned_signal", 
            level, 
            split, 
            save_dir, 
            extra_description=extra_description
        )
        self.bin_values_file_path = self.make_file_path("bin_values")
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.shuffle = shuffle

        if regenerate:
            self.generate()

    def generate(self):    

        df_agg = load_aggregated_raw_signal(
            self.level, 
            self.split, 
            self.save_dir
        )
        
        def convert_to_binned_df(df_agg):
            bins, bin_values = to_bins(df_agg[self.label_name])
            df_agg[self.binned_label_name] = bins
            df_agg = df_agg.drop(columns=self.label_name)
            return df_agg, bin_values
        df_agg, bin_values = convert_to_binned_df(df_agg)
        
        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            if self.q_squared_veto:
                df_agg = apply_q_squared_veto(df_agg)
            if self.std_scale:
                for column_name in self.feature_names:
                    df_agg[column_name] = (
                        (df_agg[column_name] - means[self.level][column_name]) 
                        / stds[self.level][column_name]
                    )
            if self.balanced_classes:
                df_agg = balance_classes(df_agg, self.binned_label_name)
            if self.shuffle:
                df_agg = df_agg.sample(frac=1)
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        features = torch.from_numpy(df_agg[self.feature_names].to_numpy())
        labels = torch.from_numpy(df_agg[self.binned_label_name].to_numpy())
        bin_values = torch.from_numpy(bin_values)

        torch.save(features, self.features_file_path)
        torch.save(labels, self.labels_file_path)
        torch.save(bin_values, self.bin_values_file_path)
        
        print(f"Generated features of shape: {features.shape}.")
        print(f"Generated labels of shape: {labels.shape}.")
        print(f"Generated bin values of shape: {bin_values.shape}.")

    def load(self):

        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        self.bin_values = torch.load(self.bin_values_file_path, weights_only=True)
        print(f"Loaded features of shape: {self.features.shape}.")
        print(f"Loaded labels of shape: {self.labels.shape}.")
        print(f"Loaded bin values of shape: {self.bin_values.shape}.")

    def unload(self):
        del self.features
        del self.labels
        del self.bin_values
        print("Unloaded data.")


class Signal_Sets_Dataset(Custom_Dataset):
    """Torch dataset of bootstrapped sets of signal events."""

    def __init__(
            self, 
            level, 
            split, 
            save_dir, 
            num_events_per_set,
            num_sets_per_label,
            binned=False,
            q_squared_veto=True,
            std_scale=True,
            balanced_classes=True,
            labels_to_sample=None,
            extra_description=None,
            regenerate=False,
    ):
        super().__init__(
            "signal_sets", 
            level, 
            split, 
            save_dir, 
            extra_description=extra_description
        )

        self.bin_values_file_path = self.make_file_path("bin_values")
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.binned = binned
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.labels_to_sample = labels_to_sample

        if regenerate:
            self.generate()

    def generate(self):
        """
        Generate and save dataset state.
        """
        label_column_name = (
            self.binned_label_name if self.binned
            else self.label_name
        )

        df_agg = load_aggregated_raw_signal(
            self.level, 
            self.split, 
            self.save_dir
        )

        def convert_to_binned_df(df_agg):
            bins, bin_values = to_bins(df_agg[self.label_name])
            df_agg[self.binned_label_name] = bins
            df_agg = df_agg.drop(columns=self.label_name)
            return df_agg, bin_values
        if self.binned:
            df_agg, bin_values = convert_to_binned_df(df_agg)
            bin_values = torch.from_numpy(bin_values)

        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            if self.q_squared_veto:
                df_agg = apply_q_squared_veto(df_agg)
            if self.std_scale:
                for column_name in self.feature_names:
                    df_agg[column_name] = (
                        (df_agg[column_name] - means[self.level][column_name]) 
                        / stds[self.level][column_name]
                    )
            if self.balanced_classes:
                df_agg = balance_classes(
                    df_agg, 
                    label_column_name=label_column_name
                )
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        source_features = torch.from_numpy(
            df_agg[self.feature_names]
            .to_numpy()
        )
        source_labels = torch.from_numpy(
            df_agg[label_column_name]
            .to_numpy()
        )

        if self.binned and self.labels_to_sample:
            self.labels_to_sample = [
                torch.argwhere(bin_values == label)
                .item() 
                for label in self.labels_to_sample
            ]

        features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=self.num_events_per_set, 
            m=self.num_sets_per_label,
            reduce_labels=True,
            labels_to_sample=self.labels_to_sample,
        )

        torch.save(
            features, 
            self.features_file_path
        )
        print(
            f"Generated features of shape: {features.shape}."
        )
        torch.save(
            labels, 
            self.labels_file_path
        )
        print(
            f"Generated labels of shape: {labels.shape}."
        )
        if self.binned:
            torch.save(
                bin_values, 
                self.bin_values_file_path
            )
            print(
               f"Generated bin values of shape: {bin_values.shape}."
            )

    def load(self): 
        """
        Load saved dataset state. 
        """
        self.features = torch.load(
            self.features_file_path, 
            weights_only=True
        )
        print(
            f"Loaded features of shape: {self.features.shape}."
        )
        self.labels = torch.load(
            self.labels_file_path, 
            weights_only=True
        )
        print(
            f"Loaded labels of shape: {self.labels.shape}."
        )
        if self.binned:
            self.bin_values = torch.load(
                self.bin_values_file_path,
                weights_only=True
            )
            print(
                f"Loaded bin values of shape: {self.bin_values.shape}."
            )

    def unload(self):
        del self.features
        del self.labels
        if self.binned:
            del self.bin_values
        print("Unloaded data.")


class Signal_Images_Dataset(Custom_Dataset):
    """Bootstrapped images (like Shawn)."""
    def __init__(
            self, 
            level, 
            split, 
            save_dir, 
            num_events_per_set,
            num_sets_per_label,
            n_bins,
            q_squared_veto=True,
            std_scale=True,
            balanced_classes=True,
            labels_to_sample=None,
            extra_description=None,
            regenerate=False,
    ):
        super().__init__(
            "signal_images", 
            level, 
            split, 
            save_dir, 
            extra_description=extra_description
        )

        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.n_bins = n_bins
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.labels_to_sample = labels_to_sample

        if regenerate:
            self.generate()
    
    def generate(self):

        df_agg = load_aggregated_raw_signal(self.level, self.split, self.save_dir)
        
        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            if self.q_squared_veto:
                df_agg = apply_q_squared_veto(df_agg)
            if self.std_scale:
                df_agg["q_squared"] = (
                    (df_agg["q_squared"] - means[self.level]["q_squared"]) 
                    / stds[self.level]["q_squared"]
                )
            if self.balanced_classes:
                df_agg = balance_classes(df_agg, label_column_name=self.label_name)
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        source_features = torch.from_numpy(df_agg[self.feature_names].to_numpy())
        source_labels = torch.from_numpy(df_agg[self.label_name].to_numpy())
        set_features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=self.num_events_per_set, m=self.num_sets_per_label,
            reduce_labels=True,
            labels_to_sample=self.labels_to_sample
        )
        features = torch.cat(
            [
                make_image(set_feat, n_bins=self.n_bins).unsqueeze(0) 
                for set_feat in set_features.numpy()
            ]
        )
        
        torch.save(features, self.features_file_path)
        torch.save(labels, self.labels_file_path)
        print(f"Generated features of shape: {features.shape}.")
        print(f"Generated labels of shape: {labels.shape}.")

    def load(self): 
        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        print(f"Loaded features of shape: {self.features.shape}.")
        print(f"Loaded labels of shape: {self.labels.shape}.")

    def unload(self):
        del self.features
        del self.labels
        print("Unloaded data.")



###############################################
###############################################
        


# class Bootstrapped_Signal_Binned_Dataset(Dataset):

#     def __init__(self, level, split, save_dir):
#         self.level = level
#         self.split = split
#         self.save_dir = pathlib.Path(save_dir)
        
#         self.feature_names = ["q_squared", "costheta_mu", "costheta_K", "chi"]
#         self.dc9_column_name = "dc9" # defined elsewhere as well: can fix this
#         self.dc9_bin_column_name = "dc9_bin_index"

#         features_file_name = self.make_features_save_file_name()
#         labels_file_name = self.make_labels_save_file_name()
#         bin_values_file_name = self.make_bin_values_save_file_name()
#         self.features_save_path = self.save_dir.joinpath(features_file_name)
#         self.labels_save_path = self.save_dir.joinpath(labels_file_name)
#         self.bin_values_save_path = self.save_dir.joinpath(bin_values_file_name)

#     def make_features_save_file_name(self):
#         name = f"boot_sig_bin_feat_{self.level}_{self.split}.pt"
#         return name

#     def make_labels_save_file_name(self):
#         name = f"boot_sig_bin_lab_{self.level}_{self.split}.pt"
#         return name

#     def make_bin_values_save_file_name(self):
#         name = f"boot_sig_bin_values_{self.level}_{self.split}.npy"
#         return name

#     def generate(
#         self, raw_trials, raw_signal_dir, 
#         num_events_per_set, num_sets_per_label, 
#         q_squared_veto=True, std_scale=True, balanced_classes=True,
#     ):
#         df_agg = aggregate_raw_signal(self.level, raw_trials, self.feature_names, raw_signal_dir)
        
#         bins, bin_values = to_bins(df_agg[self.dc9_column_name])
#         numpy.save(self.bin_values_save_path, bin_values)
        
#         df_agg[self.dc9_bin_column_name] = bins
#         df_agg = df_agg.drop(columns=self.dc9_column_name)
        
#         if q_squared_veto:
#             df_agg = apply_q_squared_veto(df_agg)

#         if std_scale:
#             for column_name in self.feature_names:
#                 df_agg[column_name] = (df_agg[column_name] - means[self.level][column_name]) / stds[self.level][column_name]

#         if balanced_classes:
#             df_agg = balance_classes(df_agg, self.dc9_bin_column_name)

#         df_sets = bootstrap_labeled_sets(
#             df_agg,
#             self.dc9_bin_column_name,
#             num_events_per_set, num_sets_per_label
#         )

#         set_labels = df_sets.index.get_level_values("label")[::num_events_per_set]
#         labels = torch.from_numpy(set_labels.to_numpy())
#         torch.save(labels, self.labels_save_path)

#         num_sets = len(labels)
#         set_indices = range(num_sets)
#         list_of_sets = [df_sets.xs(i, level="set") for i in set_indices]
#         expanded_list_of_sets = [numpy.expand_dims(s, axis=0) for s in list_of_sets]
#         ndarray_of_sets = numpy.concatenate(expanded_list_of_sets, axis=0)
#         features = torch.from_numpy(ndarray_of_sets)
#         torch.save(features, self.features_save_path)

#     def load(self, device="cpu"): 
#         self.bin_values = numpy.load(self.bin_values_save_path, allow_pickle=True)
#         self.features = torch.load(self.features_save_path, weights_only=True)
#         self.labels = torch.load(self.labels_save_path, weights_only=True)
#         print(f"Loaded data with features size: {self.features.shape} and labels size: {self.labels.shape}.")

#         if device != "cpu":
#             self.features = self.features.to(device)
#             self.labels = self.labels.to(device)

#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         x = self.features[index]
#         y = self.labels[index]
#         return x, y
    

# class Unbinned_Signal_Dataset(Custom_Dataset):
#     """Dataset of unbinned signal events."""

#     def __init__(self, name, level, split, save_dir, regenerate):
#         self.super().__init__(name, level, split, save_dir, regenerate)

#     def generate(self, raw_trials, raw_signal_dir,  
#         q_squared_veto=True, std_scale=True, balanced_classes=True,
#         dtype="float32"
#     ):
#         """Generate and save dataset state."""

#         df_agg = aggregate_raw_signal(self.level, raw_trials, self.feature_names, raw_signal_dir, dtype=dtype)
        
#         if q_squared_veto:
#             df_agg = apply_q_squared_veto(df_agg)

#         if std_scale:
#             for column_name in self.feature_names:
#                 df_agg[column_name] = (df_agg[column_name] - means[self.level][column_name]) / stds[self.level][column_name]

#         if balanced_classes:
#             df_agg = balance_classes(df_agg, self.label_name)

#         labels = df_agg[[self.label_name]]
#         features = df_agg[self.feature_names]

#         labels = torch.from_numpy(labels.to_numpy())
#         features = torch.from_numpy(features.to_numpy())

#         torch.save(labels, self.labels_file_path)
#         torch.save(features, self.features_file_path)

#     def load(self, device="cpu"):
#         """Load dataset state."""

#         self.features = torch.load(self.features_file_path, weights_only=True)
#         self.labels = torch.load(self.labels_file_path, weights_only=True)
#         if device != "cpu":
#             self.features = self.features.to(device)
#             self.labels = self.labels.to(device)
   
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         x = self.features[index]
#         y = self.labels[index]
#         return x, y


# def make_image(df, n_bins):
#     """
#     Make an image of a dataset. (Like Shawn.)

#     Parameters
#     ----------
#     df : Dataframe of dataset
#     n_bins : The number of bins of each angular variable.

#     Returns
#     -------
#     pytorch Tensor image. 
#     """
#     df = df.copy()
#     bins = {
#         "chi": numpy.linspace(start=0, stop=2*numpy.pi, num=n_bins+1),
#         "costheta_mu": numpy.linspace(start=-1, stop=1, num=n_bins+1),
#         "costheta_K": numpy.linspace(start=-1, stop=1, num=n_bins+1),
#     }
#     var_col_names = [*bins.keys()]
#     bin_col_names = [var_col_name + "_bin" for var_col_name in var_col_names]
#     for var_col_name, bin_col_name in zip(var_col_names, bin_col_names):
#         df[bin_col_name] = pandas.cut(df[var_col_name], bins=bins[var_col_name], include_lowest=True)
#     df_image = df.groupby(bin_col_names, observed=False).mean()["q_squared"]
#     numpy_image = df_image.to_numpy().reshape((n_bins,)*3) # dimensions of (chi, costheta_mu, costheta_K)
#     numpy_image = numpy.expand_dims(numpy_image, axis=0)
#     numpy_image = numpy.nan_to_num(numpy_image)
#     torch_image = torch.from_numpy(numpy_image)
#     return torch_image