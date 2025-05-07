
"""
Configuration for datasets.
"""


import pathlib


class Dataset_Config:
    """
    Dataset configuration.
    """

    def __init__(
        self,
        name:str,
        level:str,
        q_squared_veto:str,
        balanced_classes:bool,
        std_scale:bool,
        split:str,
        path_dir_parent:str|pathlib.Path,
        path_dir_raw_signal:str|pathlib.Path,
        shuffle:bool,
        label_subset:list=None, # original labels (not bin values)
        num_events_per_set:int=None,
        num_sets_per_label:int=None,
        num_bins_image:int=None,
        extra_description:str=None,
    ):
        """
        Initialization.
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
        self.path_dir_parent = pathlib.Path(path_dir_parent)
        self.path_dir_raw_signal = pathlib.Path(path_dir_raw_signal)
        self.shuffle = shuffle
        self.label_subset = label_subset
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.num_bins_image = num_bins_image
        self.extra_description = extra_description

        self.path_dir = self._make_path_dir(
            name, 
            level, 
            q_squared_veto, 
            path_dir_parent,
        )

        self.path_features = self._make_path_file_tensor(
            "features", 
        )
        self.path_labels = self._make_path_file_tensor(
            "labels",
        )            
        self.path_bin_map = self._make_path_file_tensor(
            "bin_map",       
        )

        self.trial_range_raw_signal = (
            self._convert_split_to_trial_range_raw_signal(split)
        )

    def _define_constants(self):

        self.name_dset_images_signal = "images_signal"
        self.name_dset_binned_signal = "binned_signal"
        self.name_dset_sets_binned_signal = "sets_binned_signal"
        self.name_dset_sets_unbinned_signal = "sets_unbinned_signal"

        self.names_datasets = [
            self.name_dset_images_signal,
            self.name_dset_binned_signal,
            self.name_dset_sets_binned_signal,
            self.name_dset_sets_unbinned_signal,
        ]

        self.name_level_generator = "gen"
        self.name_level_detector = "det"

        self.names_levels = (
            self.name_level_generator, 
            self.name_level_detector
        )

        self.name_q_squared_veto_tight = "tight"
        self.name_q_squared_veto_loose = "loose"
        
        self.names_q_squared_vetos = (
            self.name_q_squared_veto_tight, 
            self.name_q_squared_veto_loose
        )

        self.name_var_q_squared = "q_squared"
        self.name_var_cos_theta_mu = "costheta_mu"
        self.name_var_cos_k = "costheta_K"
        self.name_var_chi = "chi"

        self.names_features = (
            self.name_var_q_squared,
            self.name_var_cos_theta_mu,
            self.name_var_cos_k,
            self.name_var_chi,
        )

        self.name_label_unbinned = "dc9"
        self.name_label_binned = "dc9_bin_index"

        self.names_labels = (
            self.name_label_unbinned,
            self.name_label_binned,
        )
                
        self.name_split_train = "train"
        self.name_split_eval = "eval"

        self.names_dset_splits = (
            self.name_split_train,
            self.name_split_eval,
        )

        self.trials_split_train = range(1,21)
        self.trials_split_eval = range(21,41)

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
        if name not in self.names_datasets:
            raise ValueError(f"Name not recognized: {name}")
        if level not in self.names_levels:
            raise ValueError(f"Level not recognized: {level}")
        if q_squared_veto not in self.names_q_squared_vetos:
            raise ValueError(f"q^2 veto not recognized: {q_squared_veto}")
        if balanced_classes not in (True, False):
            raise ValueError("Balanced classes option not recognized.")
        if std_scale not in (True, False):
            raise ValueError("Standard scale option not recognized.")
        if split not in self.names_dset_splits:
            raise ValueError(f"Split not recognized: {split}")
        if shuffle not in (True, False):
            raise ValueError("Shuffle option not recognized.")

    def _make_path_dir(self):
        """
        Create the dataset's subdirectory path.
        """

        name_dir = f"{self.name}_{self.level}_q2v_{self.q_squared_veto}"
        path = self.path_dir_parent.joinpath(name_dir)
        return path

    def _make_path_file_tensor(self, kind):
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

        name_file = (
            f"{self.extra_description}_{self.split}_{kind}.pt" 
            if self.extra_description
            else f"{self.split}_{kind}.pt"
        )
        path = self.path_dir.joinpath(name_file)
        return path

    def _convert_split_to_trial_range_raw_signal(self):
        """
        Obtain the raw signal trial range corresponding to
        the data split.
        """
        
        if self.split not in self.names_dset_splits:
            raise ValueError(
                f"Split must be in {self.names_dset_splits}"
            )
        trial_range = (
            self.trials_split_train if (
                self.split==self.name_split_train
            )
            else self.trials_split_eval if (
                self.split==self.name_split_eval
            )
            else None
        )
        return trial_range