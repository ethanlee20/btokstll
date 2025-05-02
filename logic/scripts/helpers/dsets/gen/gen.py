

"""
Dataset generation stuff.
"""


import numpy
import torch 

from helpers.dsets.config import Config
from helpers.dsets.gen.preproc import (
    convert_to_binned,
    bootstrap_labeled_sets,
    apply_common_preprocessing
)


def print_done_status(kind, shape, path, verbose=True):
    if verbose: 
        print(
            f"Generated {kind} of shape: "
            f"{shape}."
            f"\nSaved as: {path}"
        )

def save_torch_file(verbose=True)

def generate_binned_signal(
    df_agg,
    config:Config,
    verbose:bool=True,
):
    """
    Generate files for the binned signal dataset.
    """
    
    df_agg = apply_common_preprocessing(df_agg, config)
        
    df_agg, bin_values = convert_to_binned(
        df_agg, 
        config.label_name, 
        config.binned_label_name
    )

    features = torch.from_numpy(
        df_agg[config.feature_names].to_numpy()
    )
    labels = torch.from_numpy(
        df_agg[config.binned_label_name].to_numpy()
    )
    bin_values = torch.from_numpy(bin_values)

    torch.save(features, config.features_path)
    print_done_status(
        "features", 
        features.shape, 
        config.features_path,
        verbose=verbose,
    )

    torch.save(labels, config.labels_path)
    print_done_status(
        "labels", 
        labels.shape, 
        config.labels_path
        verbose=verbose,
    )

    torch.save(bin_values, config.bin_values_path)
    print_done_status(
        "bin values", 
        bin_values.shape, 
        config.bin_values_path
        verbose=verbose,
    )

    
def generate_binned_signal_sets(
    df_agg, 
    config:Config, 
    verbose:bool=True,
):
    """
    Generate files for the 
    binned signal sets dataset.
    """    
    df_agg = df_agg.copy() 

    df_agg = apply_q_squared_veto(
        df_agg, 
        config.q_squared_veto
    )
    if config.std_scale:
        df_agg = apply_standard_scale(
            df_agg, 
            config.level, 
            config.q_squared_veto, 
            config.feature_names,
        )  
    if config.balanced_classes:
        df_agg = apply_balance_classes(
            df_agg, 
            config.label_name,
        )
    if config.label_subset:
        df_agg = apply_label_subset(
            df_agg,
            config.label_name,
            config.label_subset,
        )
    if config.shuffle:
        df_agg = df_agg.sample(frac=1)

    df_agg, bin_values = convert_to_binned(
        df_agg, 
        config.label_name, 
        config.binned_label_name,
    )
    
    bin_values = torch.from_numpy(bin_values)
    source_features = torch.from_numpy(
        df_agg[config.feature_names]
        .to_numpy()
    )
    source_labels = torch.from_numpy(
        df_agg[config.label_column_name]
        .to_numpy()
    )

    features, labels = bootstrap_labeled_sets(
        source_features,
        source_labels,
        n=config.num_events_per_set, 
        m=config.num_sets_per_label,
        reduce_labels=True,
    )

    torch.save(features, config.features_path)
    torch.save(labels, config.labels_path)
    torch.save(bin_values, config.bin_values_path)

    if verbose:
        print(
            "Generated features of shape: "
            f"{features.shape}."
            f"\nSaved as: {config.features_path}"
        )
        print(
            "Generated labels of shape: "
            f"{labels.shape}."
            f"\nSaved as: {config.labels_path}"
        )
        print(
            "Generated bin values of shape: "
            f"{bin_values.shape}."
            f"\nSaved as: {config.bin_values_path}"
        )

def generate_unbinned_signal_sets(df_agg):
    pass

def _generate_signal_images(df_agg):
    pass





class Dataset_Generator:
    """
    Generates (and saves) a dataset. 
    """
    def __init__(
        self, 
        config:Config,
    ):
        self.config = config
        self._check_files_do_not_exist()
        self._load_agg_raw_signal_data()
        if config.name == "binned_signal":
            _generate_binned_signal()
        elif config.name == "binned_signal_sets":
            _generate_binned_signal_sets()
        elif config.name == "unbinned_signal_sets":
            self._generate_unbinned_signal_sets()
        elif config.name == "signal_images":
            self._generate_signal_images()
        else: raise ValueError(
            f"{config.name} not recognized."
        )

    def _load_agg_raw_signal_data(self):
        """
        Load the aggregated raw signal data.
        Generate the aggregated raw signal data file
        if it doesn't already exist.
        """
        if not make_aggregated_raw_signal_file_save_path(
            self.config.dir_path,
            self.config.level,
            self.config.raw_signal_trial_range,
        ).is_file():
            print(
                "Aggregated raw signal file not found, "
                "generating..."
            )
            aggregate_raw_signal_data_files(
                self.config.level, 
                self.config.raw_signal_trial_range, 
                self.config.feature_names,
                self.config.raw_signal_dir_path,
                self.config.dir_path,
            )

        self.df_agg = load_aggregated_raw_signal_data_file(
            self.save_dir,
            self.level, 
            self.raw_signal_trial_range, 
        )
    
    def _check_files_do_not_exist(self):
        [
            self._assert_file_does_not_exist(path)
            for path in [
                self.config.features_path, 
                self.config.labels_path,
                self.config.bin_values_path,
            ]
        ]
        self._load_agg_raw_signal_data()

    def _assert_file_does_not_exist(self, path:pathlib.Path):
        """
        Assert that file does not exist.

        Parameters
        ----------
        path : pathlib.Path
            The file's path.
        """
        path = pathlib.Path(path)
        if path.is_file():
            raise ValueError(
                f"Attempting to generate: {path}, "
                "but file already exists." 
                "Delete and rerun to proceed."
            )