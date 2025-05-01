
import numpy

from helpers.dsets.config import Config






def _generate_binned_signal(df_agg):
    
        
        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            df_agg = apply_q_squared_veto(
                df_agg, 
                self.q_squared_veto
            )
            if self.std_scale:
                for column_name in self.feature_names:
                    df_agg[column_name] = ( 
                        (
                            df_agg[column_name] 
                            - get_dataset_prescale(
                                "mean", 
                                self.level, 
                                self.q_squared_veto, 
                                column_name
                            )
                        ) 
                        / get_dataset_prescale(
                            "std", 
                            self.level, 
                            self.q_squared_veto, 
                            column_name
                        )
                    )
            if self.balanced_classes:
                df_agg = balance_classes(
                    df_agg, 
                    self.binned_label_name
                )
            if self.shuffle:
                df_agg = df_agg.sample(frac=1)
            return df_agg
        
        df_agg, bin_values = convert_to_binned(self.df_agg)
        df_agg = apply_preprocessing(df_agg)

        features = torch.from_numpy(
            df_agg[self.feature_names].to_numpy()
        )
        labels = torch.from_numpy(
            df_agg[self.binned_label_name].to_numpy()
        )
        bin_values = torch.from_numpy(bin_values)

        torch.save(features, self.features_path)
        torch.save(labels, self.labels_path)
        torch.save(bin_values, self.bin_values_path)
        
        print(f"Generated features of shape: {features.shape}.")
        print(f"Generated labels of shape: {labels.shape}.")
        print(f"Generated bin values of shape: {bin_values.shape}.")

def _generate_binned_signal_sets(df_agg):
    pass

def _generate_unbinned_signal_sets(df_agg):
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