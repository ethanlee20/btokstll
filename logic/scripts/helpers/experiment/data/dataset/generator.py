
import pathlib

import torch 

from .config import Config
from .preproc import (
    convert_to_binned, 
    apply_cleaning, 
    bootstrap_labeled_sets,
    pandas_to_torch
)
from ..file_hand import (
    load_aggregated_raw_signal_data_file, 
    make_aggregated_raw_signal_file_save_path,
    aggregate_raw_signal_data_files,
    save_file_torch_tensor
)


class Dataset_Generator:
    """
    Generates a dataset. 
    """
    def __init__(
        self, 
        config:Config,
    ):
        """
        Initialize.
        """

        self.config = config
        self._check_files_do_not_exist()
        self._load_agg_signal_data()

    def _load_agg_signal_data(self):
        """
        Load the aggregated raw signal data.
        Generate the aggregated raw signal data file
        if it doesn't already exist.
        """

        if not make_aggregated_raw_signal_file_save_path(
            self.config.dir_path_dataset,
            self.config.level,
            self.config.trial_range_raw_signal,
        ).is_file():
            print(
                "Aggregated raw signal file not found, "
                "generating..."
            )
            aggregate_raw_signal_data_files(
                self.config.level, 
                self.config.trial_range_raw_signal, 
                self.config.names_features,
                self.config.dir_path_raw_signal,
                self.config.dir_path_dataset,
            )

        df_agg = load_aggregated_raw_signal_data_file(
            self.save_dir,
            self.level, 
            self.raw_signal_trial_range, 
        )

        self.df_agg = apply_cleaning(
            df_agg, 
            self.config,
        )

    def _check_files_do_not_exist(self):
        """
        Check labels, features, and bin map files
        do not exist.
        """

        def assert_file_does_not_exist(
            path:pathlib.Path|str
        ):
            path = pathlib.Path(path)
            if path.is_file():
                raise ValueError(
                    f"File: {path} already exists. " 
                    "Delete the file to proceed."
                ) 
        [
            assert_file_does_not_exist(path)
            for path in [
                self.config.path_features, 
                self.config.path_labels,
                self.config.path_bin_map,
            ]
        ]

    def _generate_binned_signal(self):
        """
        Generate files for the 
        binned signal dataset.
        """

        config = self.config
        df_agg = self.df_agg.copy()
            
        df_agg, bin_map = convert_to_binned(
            df_agg, 
            config.name_label_unbinned, 
            config.name_label_binned
        )

        features = pandas_to_torch(
            df_agg[config.names_features]
        )
        labels = pandas_to_torch(
            df_agg[config.name_label_binned]
        )
        bin_map = torch.from_numpy(bin_map)

        save_file_torch_tensor(
            features, 
            config.path_features,
        )
        save_file_torch_tensor(
            labels, 
            config.path_labels,
        )
        save_file_torch_tensor(
            bin_map, 
            config.path_bin_map,
        )

    def _generate_sets_binned_signal(self):
        """
        Generate files for the 
        sets binned signal dataset.
        """    

        config = self.config
        df_agg = self.df_agg.copy()

        df_agg, bin_map = convert_to_binned(
            df_agg, 
            config.name_label_unbinned, 
            config.name_label_binned,
        )
        
        source_features = pandas_to_torch(
            df_agg[config.names_features]
        )
        source_labels = pandas_to_torch(
            df_agg[config.name_label_binned]
        )
        bin_map = torch.from_numpy(bin_map)

        features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=config.num_events_per_set, 
            m=config.num_sets_per_label,
            reduce_labels=True,
        )

        save_file_torch_tensor(
            features, 
            config.path_features
        )
        save_file_torch_tensor(
            labels, 
            config.path_labels
        )
        save_file_torch_tensor(
            bin_map, 
            config.path_bin_map
        )

    def _generate_sets_unbinned_signal(self):
        """
        Generate files for the
        sets unbinned signal dataset.
        """

        config = self.config
        df_agg = self.df_agg.copy()

        source_features = pandas_to_torch(
            df_agg[config.names_features]
        )
        source_labels = pandas_to_torch(
            df_agg[config.name_label_unbinned]
        )
        
        features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=config.num_events_per_set,
            m=config.num_sets_per_label,
            reduce_labels=True,
        )

        save_file_torch_tensor(
            features, 
            config.path_features
        )
        save_file_torch_tensor(
            labels, 
            config.path_labels
        )


    def generate_signal_images():
        pass



