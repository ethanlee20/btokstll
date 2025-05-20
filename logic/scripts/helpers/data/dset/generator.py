
import pathlib

import pandas
import torch 

from .config import Config_Dataset
from .constants import Names_Levels
from .preproc import (
    convert_to_binned, 
    bins_to_probs,
    bkg_probs,
    apply_cleaning_signal, 
    apply_cleaning_bkg,
    bootstrap_labeled_sets,
    bootstrap_bkg,
    pandas_to_torch,
    make_image
)
from ..file_hand import (
    load_file_agg_raw_signal,
    load_file_raw_bkg, 
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
        config:Config_Dataset,
    ):
        """
        Initialize.
        """

        self.config = config

        self._check_files_do_not_exist()

        self._load_agg_signal_data()

        if config.level == (
            Names_Levels()
            .detector_and_background
        ):
            self._load_bkg_data()
            
        self._make_dir()

    def generate(self):
        """
        Generate dataset.
        """
        
        config = self.config

        if config.name == config.name_dset_binned_signal:

            self._generate_binned_events()

        elif config.name == config.name_dset_sets_binned_signal:

            self._generate_sets_binned()

        elif config.name == config.name_dset_sets_unbinned_signal:
            
            self._generate_sets_unbinned_signal()

        elif config.name == config.name_dset_images_signal:

            self._generate_images_signal()

        else:

            raise ValueError(f"Name not recognized: {config.name}")
        
        print(f"Generated dataset: {config.name}")

    def _generate_binned_events(self):

        """
        Generate files for the 
        binned events dataset.
        """

        config = self.config

        df_agg = self.df_agg.copy()
        
        df_agg, bin_map = convert_to_binned(
            df_agg, 
            config.name_label_unbinned, 
            config.name_label_binned
        )
        
        features_signal = pandas_to_torch(
            df_agg[config.names_features]
        )

        labels_signal = pandas_to_torch(
            bins_to_probs(
                df_agg[config.name_label_binned]
            )
        )

        bin_map = torch.from_numpy(bin_map)

        if config.level == (
            Names_Levels()
            .detector_and_background
        ):

            df_bkg = pandas.concat(
                self.df_bkg_charge,
                self.df_bkg_mix,
            )

            features_bkg = pandas_to_torch(
                df_bkg[config.names_features]
            )

            labels_bkg = bkg_probs(
                num_events=len(features_bkg),
                num_bins=labels_signal.shape[1]
            )

            features = torch.concat(
                [
                    features_signal, 
                    features_bkg,
                ]
            )

            labels = torch.concat(
                [
                    labels_signal, 
                    labels_bkg
                ]
            )

        else:

            features = features_signal

            labels = labels_signal

        save_file_torch_tensor(
            features, 
            config.path_file_features,
        )

        save_file_torch_tensor(
            labels, 
            config.path_file_labels,
        )

        save_file_torch_tensor(
            bin_map, 
            config.path_file_bin_map,
        )

    def _generate_sets_binned(self):

        """
        Generate files for the 
        sets binned dataset.
        """    

        config = self.config

        df_agg = self.df_agg.copy()

        df_agg, bin_map = convert_to_binned(
            df_agg, 
            config.name_label_unbinned, 
            config.name_label_binned,
        )
        
        source_features_signal = pandas_to_torch(
            df_agg[config.names_features]
        )

        source_labels_signal = df_agg[config.name_label_binned]

        bin_map = torch.from_numpy(bin_map)

        features_signal, labels = (
            bootstrap_labeled_sets(
                source_features_signal,
                source_labels_signal,
                num_events_per_set=config.num_events_per_set_signal, 
                num_sets_per_label=config.num_sets_per_label,
                reduce_labels=True,
            )
        )

        labels = pandas_to_torch(
            bins_to_probs(labels)
        )

        if config.level == (
            Names_Levels().detector_and_background
        ):
            
            num_labels = labels.shape[1]

            num_sets = config.num_sets_per_label * num_labels
            
            features_bkg = bootstrap_bkg(
                self.df_bkg_charge, 
                self.df_bkg_mix, 
                config.num_events_per_set, 
                num_sets, 
                frac_charge=0.5,
            )

            features = torch.concat(
                [
                    features_signal,
                    features_bkg,
                ],
                dim=1,
            )

        else:
            features = features_signal

        save_file_torch_tensor(
            features, 
            config.path_file_features
        )

        save_file_torch_tensor(
            labels, 
            config.path_file_labels
        )

        save_file_torch_tensor(
            bin_map, 
            config.path_file_bin_map
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
            num_events_per_set=config.num_events_per_set,
            num_sets_per_label=config.num_sets_per_label,
            reduce_labels=True,
        )

        save_file_torch_tensor(
            features, 
            config.path_file_features
        )

        save_file_torch_tensor(
            labels, 
            config.path_file_labels
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
                num_events_per_set=config.num_events_per_set,
                num_sets_per_label=config.num_sets_per_label,
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
            config.path_file_features
        )

        save_file_torch_tensor(
            labels, 
            config.path_file_labels
        )

    def _load_agg_signal_data(self):

        """
        Load the aggregated raw signal data.
        Generate the aggregated raw signal 
        data file if it doesn't already exist.
        """

        config = self.config

        if not make_path_file_agg_raw_signal(
            config.path_dir_dsets_main,
            config.level,
            config.range_trials_raw_signal,
        ).is_file():
            
            print(
                "Aggregated raw signal file "
                "not found, generating..."
            )

            agg_data_raw_signal(
                config.level, 
                config.range_trials_raw_signal, 
                config.names_features,
                config.path_dir_raw_signal,
                config.path_dir_dsets_main,
            )

        df_agg = load_file_agg_raw_signal(
            config.path_dir_dsets_main,
            config.level, 
            config.range_trials_raw_signal, 
        )

        self.df_agg = apply_cleaning_signal(
            df_agg, 
            self.config,
        )

    def _load_bkg_data(self):
        
        df_bkg_charge, df_bkg_mix = [
            load_file_raw_bkg(
                dir=self.config.path_dir_raw_bkg,
                charge_or_mix=kind,
                split=self.config.split,
            )
            for kind in ("charge", "mix")
        ]

        self.df_bkg_charge, self.df_bkg_mix = [
            apply_cleaning_bkg(
                df,
                self.config,
            )
            for df in (df_bkg_charge, df_bkg_mix)
        ]

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
                config.path_file_features, 
                config.path_file_labels,
                config.path_file_bin_map,
            ]
        ]

    def _make_dir(self):

        """
        Make the dataset's directory.
        """

        config = self.config
        
        config.path_dir.mkdir(exist_ok=True)

