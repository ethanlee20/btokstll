
import pathlib

import numpy
import torch 

from .config import Config


class Dataset_Generator:
    """
    Generates a dataset. 
    """
    def __init__(
        self, 
        config:Config,
    ):
        self.config = config
        self._check_files_do_not_exist()
        self._load_agg_raw_signal_data()

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
        """
        Check labels, features, and bin values files
        do not exist.
        """
        def assert_file_does_not_exist(path:pathlib.Path):
            path = pathlib.Path(path)
            if path.is_file():
                raise ValueError(
                    f"File: {path} already exists. " 
                    "Delete the file to proceed."
                ) 
        [
            assert_file_does_not_exist(path)
            for path in [
                self.config.features_path, 
                self.config.labels_path,
                self.config.bin_values_path,
            ]
        ]

    def _save_torch_file(
        self,
        tensor:torch.Tensor, 
        path:str|pathlib.Path, 
        verbose:bool=True,
    ):
        def print_done_status(shape, path):
            print(
                f"Generated tensor of shape: "
                f"{shape}."
                f"\nSaved as: {path}"
            )
        torch.save(tensor, path)    
        if verbose:
            print_done_status(tensor.shape, path)

    def generate_signal_images():
        pass

    def generate_binned_signal(
        df_agg,
        config:Config,
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
            df_agg[config.feature_names]
            .to_numpy()
        )
        labels = torch.from_numpy(
            df_agg[config.binned_label_name]
            .to_numpy()
        )
        bin_values = torch.from_numpy(bin_values)

        _save_torch_file(
            features, 
            config.features_path
        )
        _save_torch_file(
            labels, 
            config.labels_path
        )
        _save_torch_file(
            bin_values, 
            config.bin_values_path
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



