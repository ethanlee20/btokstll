
import pathlib

import pandas
import torch 

from .config import Config_Dataset
from .constants import (
    Names_Levels, 
    Names_Datasets, 
    Names_Variables,
    Names_Labels,
)
from .preproc import (
    convert_to_binned, 
    bins_to_probs,
    bkg_probs,
    apply_cleaning_signal, 
    apply_cleaning_bkg,
    bootstrap_labeled_sets,
    bootstrap_bkg,
    torch_from_pandas,
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

        self._load_signal()

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

        if self.config.name == Names_Datasets().events_binned:
            self._generate_events_binned()

        elif self.config.name == Names_Datasets().sets_binned:
            self._generate_sets_binned()

        elif self.config.name == Names_Datasets().sets_unbinned:
            self._generate_sets_unbinned()

        elif self.config.name == Names_Datasets().images:
            self._generate_images()

        else:
            raise ValueError(
                f"Name not recognized: {self.config.name}"
            )
        
        print(f"Generated dataset: {self.config.name}")

    def _generate_events_binned(self):

        """
        Generate files for the 
        binned events dataset.
        """
        
        features_signal = torch_from_pandas(
            self.df_signal[Names_Variables().list_]
        )

        labels_signal = bins_to_probs(
            bins=torch_from_pandas(
                self.df_signal[self.config.name_label]
            ),
            num_bins=self.num_labels_unique_preclean,
        )

        if self.config.level == (
            Names_Levels()
            .detector_and_background
        ):

            df_bkg = pandas.concat(
                [
                    self.df_bkg_charge,
                    self.df_bkg_mix,
                ]
            )

            features_bkg = torch_from_pandas(
                df_bkg[Names_Variables().list_]
            )

            labels_bkg = bkg_probs(
                num_events=len(features_bkg),
                num_bins=self.num_labels_unique_preclean,
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

        num_examples = len(labels)
        index_shuffled = torch.randperm(num_examples)
        features = features[index_shuffled]
        labels = labels[index_shuffled]

        save_file_torch_tensor(
            features, 
            self.config.path_file_features,
        )

        save_file_torch_tensor(
            labels, 
            self.config.path_file_labels,
        )

        save_file_torch_tensor(
            self.bin_map, 
            self.config.path_file_bin_map,
        )

    def _generate_sets_binned(self):

        """
        Generate files for the 
        sets binned dataset.
        """    

        features, labels = self._make_sets()

        labels = bins_to_probs(
            bins=labels, 
            num_bins=self.num_labels_unique_preclean,
        )

        save_file_torch_tensor(
            self.bin_map, 
            self.config.path_file_bin_map
        )

        save_file_torch_tensor(
            features, 
            self.config.path_file_features
        )

        save_file_torch_tensor(
            labels, 
            self.config.path_file_labels
        )

    def _generate_sets_unbinned(self):

        """
        Generate files for the
        sets unbinned dataset.
        """

        features, labels = self._make_sets()

        save_file_torch_tensor(
            features, 
            self.config.path_file_features
        )

        save_file_torch_tensor(
            labels, 
            self.config.path_file_labels
        )

    def _generate_images(self):

        """
        Generate files for the 
        images dataset.
        """

        features_sets, labels = self._make_sets()

        features = torch.cat(
            [
                make_image(
                    set_,
                    n_bins=self.config.num_bins_image
                ).unsqueeze(dim=0)
                for set_ 
                in features_sets
            ]
        )

        save_file_torch_tensor(
            features, 
            self.config.path_file_features
        )

        save_file_torch_tensor(
            labels, 
            self.config.path_file_labels
        )

    def _make_sets(
        self,
    ):
        
        labels_source_signal = torch_from_pandas(
            self.df_signal[self.config.name_label]
        )
        
        features_source_signal = torch_from_pandas(
            self.df_signal[Names_Variables().list_]
        )

        features_sets_signal, labels = (
            bootstrap_labeled_sets(
                features_source_signal,
                labels_source_signal,
                num_events_per_set=self.config.num_events_per_set_signal,
                num_sets_per_label=self.config.num_sets_per_label,
                reduce_labels=True,
            )
        )

        if self.config.level == (
            Names_Levels().detector_and_background
        ):  

            num_sets_total = (
                self.config.num_sets_per_label 
                * self.num_labels_unique_postclean
            )
            
            features_sets_bkg = bootstrap_bkg(
                self.df_bkg_charge, 
                self.df_bkg_mix, 
                self.config.num_events_per_set_bkg, 
                num_sets_total, 
                frac_charge=0.5,
            )

            features = torch.concat(
                [
                    features_sets_signal,
                    features_sets_bkg,
                ],
                dim=1,
            )

        else:
            features = features_sets_signal

        return features, labels

    def _load_signal(self):

        """
        Load the aggregated raw signal data.
        Generate the aggregated raw signal 
        data file if it doesn't already exist.
        """

        level = (
            Names_Levels().detector
            if self.config.level == Names_Levels().detector_and_background
            else self.config.level
        )

        if not make_path_file_agg_raw_signal(
            dir=self.config.path_dir_dsets_main,
            level=level,
            trials=self.config.range_trials_raw_signal,
        ).is_file():
            
            print(
                "Aggregated raw signal file "
                "not found, generating..."
            )

            agg_data_raw_signal(
                level=level, 
                trials=self.config.range_trials_raw_signal, 
                columns=Names_Variables().list_,
                raw_signal_data_dir=self.config.path_dir_raw_signal,
                save_dir=self.config.path_dir_dsets_main,
            )

        df_signal = load_file_agg_raw_signal(
            dir=self.config.path_dir_dsets_main,
            level=level, 
            trials=self.config.range_trials_raw_signal, 
        )

        if self.config.is_binned:

            df_signal, self.bin_map = convert_to_binned(
                df_signal, 
                Names_Labels().unbinned, 
                Names_Labels().binned,
            )

        self.num_labels_unique_preclean = len(
            df_signal[self.config.name_label]
            .unique()
        )
        
        self.df_signal = apply_cleaning_signal(
            df_signal, 
            self.config,
            bin_map=self.bin_map
        )

        self.num_labels_unique_postclean = len(
            self.df_signal[self.config.name_label]
            .unique()
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

