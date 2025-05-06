
"""
Configuration for datasets.
"""


import pathlib


class Config:
    def __init__(
        self,
        name:str,
        level:str,
        q_squared_veto:str,
        balanced_classes:bool,
        std_scale:bool,
        split:str,
        dir_path:str|pathlib.Path,
        raw_signal_dir_path:str|pathlib.Path,
        shuffle:bool,
        label_subset:list=None, # original labels (not bin values)
        num_events_per_set:int=None,
        num_sets_per_label:int=None,
        extra_description:str=None,
    ):
        """
        Dataset configuration.
        """

        self._define_constants()
        self._check_inputs(
            name,
            level,
            q_squared_veto,
            balanced_classes,
            std_scale,
            split,
            shuffle,
        )
        
        self.name = name
        self.level = level
        self.q_squared_veto = q_squared_veto
        self.balanced_classes = balanced_classes
        self.std_scale = std_scale
        self.split = split
        self.dir_path = pathlib.Path(dir_path)
        self.raw_signal_dir_path = raw_signal_dir_path
        self.shuffle = shuffle
        self.label_subset = label_subset
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.extra_description = extra_description

        self.sub_dir_path = self._make_sub_dir_path(
            name, 
            level, 
            q_squared_veto, 
            dir_path,
        )
        self.features_path = self._make_tensor_filepath(
            "features", 
            extra_description, 
            split, 
            self.sub_dir_path,
        )
        self.labels_path = self._make_tensor_filepath(
            "labels",
            extra_description, 
            split, 
            self.sub_dir_path,
        )            
        self.bin_values_path = self._make_tensor_filepath(
            "bin_values",       
            extra_description, 
            split, 
            self.sub_dir_path,
        )

        self.raw_signal_trial_range = (
            self._convert_split_to_raw_signal_trial_range(split)
        )

    def _define_constants(self):

        self.name_images_signal_dset = "images_signal"
        self.name_binned_signal_dset = "binned_signal"
        self.name_sets_binned_signal_dset = "sets_binned_signal"
        self.name_sets_unbinned_signal_dset = "sets_unbinned_signal"

        self.dataset_names = [
            self.name_images_signal_dset,
            self.name_binned_signal_dset,
            self.name_sets_binned_signal_dset,
            self.name_sets_unbinned_signal_dset,
        ]

        self.name_generator_level = "gen"
        self.name_detector_level = "det"

        self.level_names = (
            self.name_generator_level, 
            self.name_detector_level
        )

        self.name_q_squared_veto_tight = "tight"
        self.name_q_squared_veto_loose = "loose"
        
        self.q_squared_veto_names = (
            self.name_q_squared_veto_tight, 
            self.name_q_squared_veto_loose
        )

        self.name_q_squared_var = "q_squared"
        self.name_cos_theta_mu_var = "costheta_mu"
        self.name_cos_k_var = "costheta_K"
        self.name_chi_var = "chi"

        self.feature_names = (
            self.name_q_squared_var,
            self.name_cos_theta_mu_var,
            self.name_cos_k_var,
            self.name_chi_var,
        )

        self.name_label = "dc9"
        self.name_binned_label = "dc9_bin_index"

        self.label_names = (
            self.name_label,
            self.name_binned_label,
        )
                
        self.name_train_split = "train"
        self.name_eval_split = "eval"

        self.dset_split_names = (
            self.name_train_split,
            self.name_eval_split,
        )

        self.trials_train_split = range(1,21)
        self.trials_eval_split = range(21,41)

    def _check_inputs(
        self,
        name, 
        level, 
        q_squared_veto, 
        balanced_classes, 
        std_scale, 
        split, 
        shuffle,
    ):
        if name not in self.dataset_names:
            raise ValueError(f"Name not recognized: {name}")
        if level not in self.level_names:
            raise ValueError(f"Level not recognized: {level}")
        if q_squared_veto not in self.q_squared_veto_names:
            raise ValueError(f"q^2 veto not recognized: {q_squared_veto}")
        if balanced_classes not in (True, False):
            raise ValueError("Balanced classes option not recognized.")
        if std_scale not in (True, False):
            raise ValueError("Standard scale option not recognized.")
        if split not in self.dset_split_names:
            raise ValueError(f"Split not recognized: {split}")
        if shuffle not in (True, False):
            raise ValueError("Shuffle option not recognized.")

    def _make_sub_dir_path(self):
        """
        Create the dataset's subdirectory path.
        """
        file_name = f"{self.name}_{self.level}_q2v_{self.q_squared_veto}"
        path = self.dir_path.joinpath(file_name)
        return path

    def _make_tensor_filepath(self):
        """
        Make a filepath for a torch tensor file.

        Parameters
        ----------
        kind : str
            The kind of tensor being saved. e.g. "labels".
            This determines the file name.
        
        Returns
        -------
        path : pathlib.Path
        """
        file_name = (
            f"{self.extra_description}_{self.split}_{self.kind}.pt" 
            if self.extra_description
            else f"{self.split}_{self.kind}.pt"
        )
        path = self.sub_dir_path.joinpath(file_name)
        return path

    def _convert_split_to_raw_signal_trial_range(self):
        """
        Obtain the raw signal trial range corresponding to
        the data split.
        """
        if self.split not in self.dset_split_names:
            raise ValueError(
                f"Split must be in {self.dset_split_names}"
            )
        trial_range = (
            self.trials_train_split if (
                self.split==self.name_train_split
            )
            else self.trials_eval_split if (
                self.split==self.name_eval_split
            )
            else None
        )
        return trial_range