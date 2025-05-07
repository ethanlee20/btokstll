
import pathlib

import torch 

from .config import Dataset_Config
from .preproc import (
    convert_to_binned, 
    apply_cleaning, 
    bootstrap_labeled_sets,
    pandas_to_torch,
    make_image
)
from ..file_hand import (
    load_file_agg_raw_signal, 
    make_path_file_agg_raw_signal,
    agg_data_raw_signal,
    save_file_torch_tensor
)


class Dataset_Generator:
    """
    Generates a dataset. 
    """
    def __init__(
        self, 
        config:Dataset_Config,
    ):
        """
        Initialize.
        """

        self.config = config
        self._check_files_do_not_exist()
        self._load_agg_signal_data()
        self._make_dir()

    def generate(self):
        """
        Generate dataset.
        """
        
        config = self.config
        if config.name == config.name_dset_binned_signal:
            self._generate_binned_signal()
        elif config.name == config.name_dset_sets_binned_signal:
            self._generate_sets_binned_signal()
        elif config.name == config.name_dset_sets_unbinned_signal:
            self._generate_sets_unbinned_signal()
        elif config.name == config.name_dset_images_signal:
            self._generate_images_signal()
        else:
            raise ValueError(f"Name not recognized: {config.name}")
        
        print(f"Generated dataset: {config.name}")

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

    def _generate_images_signal(self):
        """
        Generate files for the 
        images signal dataset.
        """

        config = self.config
        df_agg = self.df_agg.copy()

        features_source = pandas_to_torch(
            df_agg[config.names_features]
        )
        labels_source = pandas_to_torch(
            df_agg[config.name_label_unbinned]
        )

        features_sets_source, labels = (
            bootstrap_labeled_sets(
                features_source,
                labels_source,
                n=config.num_events_per_set,
                m=config.num_sets_per_label,
                reduce_labels=True,
            )
        )

        features = torch.cat(
            [
                make_image(
                    features_set,
                    n_bins=config.num_bins_image
                ).unsqueeze(dim=0)
                for features_set 
                in features_sets_source
            ]
        )

        save_file_torch_tensor(
            features, 
            config.path_features
        )
        save_file_torch_tensor(
            labels, 
            config.path_labels
        )

    def _load_agg_signal_data(self):
        """
        Load the aggregated raw signal data.
        Generate the aggregated raw signal 
        data file if it doesn't already exist.
        """

        config = self.config

        if not make_path_file_agg_raw_signal(
            config.path_dir_parent,
            config.level,
            config.trial_range_raw_signal,
        ).is_file():
            print(
                "Aggregated raw signal file "
                "not found, generating..."
            )
            agg_data_raw_signal(
                config.level, 
                config.trial_range_raw_signal, 
                config.names_features,
                config.path_dir_raw_signal,
                config.path_dir_parent,
            )

        df_agg = load_file_agg_raw_signal(
            config.path_dir_parent,
            config.level, 
            config.trial_range_raw_signal, 
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
            
        config = self.config

        [
            assert_file_does_not_exist(path)
            for path in [
                config.path_features, 
                config.path_labels,
                config.path_bin_map,
            ]
        ]

    def _make_dir(self):
        """
        Make the dataset's directory.
        """
        config = self.config
        config.path_dir.mkdir(exist_ok=True)

