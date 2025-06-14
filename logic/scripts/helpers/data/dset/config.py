
"""
Configuration for datasets.
"""

import pathlib

from .constants import (
    Names_Datasets,
    Names_Levels,
    Names_q_Squared_Vetos,
    Names_Splits,
    Names_Labels,
    Trials_Splits,
    Names_Kind_File_Tensor
)


class Config_Dataset:

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
        path_dir_dsets_main:str|pathlib.Path,
        path_dir_raw_signal:str|pathlib.Path,
        shuffle:bool,
        path_dir_raw_bkg:str|pathlib.Path=None,
        label_subset:list[float]=None, # original labels (not bin values)
        num_events_per_set:int=None,
        frac_bkg:float=None,
        num_sets_per_label:int=None,
        num_bins_image:int=None,
        sensitivity_study:bool=False,
    ):
        self.name = name
        self.level = level
        self.q_squared_veto = q_squared_veto
        self.balanced_classes = balanced_classes
        self.std_scale = std_scale
        self.split = split
        self.path_dir_dsets_main = pathlib.Path(
            path_dir_dsets_main
        )
        self.path_dir_raw_signal = pathlib.Path(
            path_dir_raw_signal
        )
        if path_dir_raw_bkg:
            self.path_dir_raw_bkg = pathlib.Path(
                path_dir_raw_bkg
            )
        self.shuffle = shuffle
        self.label_subset = label_subset
        self.num_events_per_set = num_events_per_set
        self.frac_bkg = frac_bkg
        self.num_sets_per_label = num_sets_per_label
        self.num_bins_image = num_bins_image
        self.sensitivity_study = sensitivity_study
        
        self._check_inputs()
        self._set_range_trials_raw_signal()
        self._set_path_dir()
        self._set_paths_files()
        self._set_is_binned()
        self._set_name_label()
        if self.num_events_per_set:
            self._calc_num_signal_bkg()

    def _check_inputs(self):

        if self.name not in Names_Datasets().tuple_:
            raise ValueError(
                f"Name not recognized: {self.name}"
            )
        
        if self.level not in Names_Levels().tuple_:
            raise ValueError(
                f"Level not recognized: {self.level}"
            )
        
        if self.q_squared_veto not in Names_q_Squared_Vetos().tuple_:
            raise ValueError(
                f"q^2 veto not recognized: {self.q_squared_veto}"
            )
        
        if self.balanced_classes not in (True, False):
            raise ValueError(
                "Balanced classes option not recognized."
            )
        
        if self.std_scale not in (True, False):
            raise ValueError(
                "Standard scale option not recognized."
            )
        
        if self.split not in Names_Splits().tuple_:
            raise ValueError(
                f"Split not recognized: {self.split}"
            )
        
        if self.shuffle not in (True, False):
            raise ValueError(
                "Shuffle option not recognized."
            )
        
        if self.frac_bkg:
            if (self.frac_bkg > 1) or (self.frac_bkg < 0):
                raise ValueError(
                    "frac_bkg must be between 0 and 1."
                )

    def _set_path_dir(self):

        """
        Create the dataset's subdirectory path.
        """

        name_dir = (
            f"{self.name}_"
            f"{self.level}_"
            f"q2v_{self.q_squared_veto}"
        )

        self.path_dir = (
            self.path_dir_dsets_main
            .joinpath(name_dir)
        )

    def _make_path_file_tensor(self, kind):

        """
        Make a filepath for a torch tensor file.
        """
        
        if kind not in Names_Kind_File_Tensor().tuple_:
            raise ValueError(
                "Kind not recognized. "
                f"Must be in {Names_Kind_File_Tensor().tuple_}"
            )
        
        name_file = (
            f"{self.split}_{kind}.pt" 
            if not self.sensitivity_study
            else f"{self.split}_sens_{kind}.pt"
        )
        if self.num_events_per_set:
            name_file = (
                f"{self.num_events_per_set}_" 
                + name_file
            )

        path = self.path_dir.joinpath(name_file)
        return path

    def _set_paths_files(self):

        self.path_file_features = self._make_path_file_tensor(
            Names_Kind_File_Tensor().features, 
        )
        self.path_file_labels = self._make_path_file_tensor(
            Names_Kind_File_Tensor().labels,
        )            
        self.path_file_bin_map = self._make_path_file_tensor(
            Names_Kind_File_Tensor().bin_map,       
        )

    def _set_range_trials_raw_signal(self):

        """
        Obtain the raw signal trial range 
        corresponding to the data split.
        """

        if self.split not in Names_Splits().tuple_:
            raise ValueError(
                f"Split must be in {Names_Splits().tuple_}"
            )
        
        self.range_trials_raw_signal = (
            Trials_Splits().train if (
                self.split == Names_Splits().train
            )
            else Trials_Splits().eval_ if (
                self.split == Names_Splits().eval_
            )
            else None
        )

    def _calc_num_signal_bkg(self):

        self.num_events_per_set_bkg = (
            int(
                self.num_events_per_set 
                * self.frac_bkg
            ) if (
                (self.frac_bkg is not None) 
                and (self.level == Names_Levels().detector_and_background)
            ) else 0
        )
        
        self.num_events_per_set_signal = (
            self.num_events_per_set 
            - self.num_events_per_set_bkg
        )

    def _set_is_binned(self):
        
        self.is_binned = (
            True 
            if self.name in (
                Names_Datasets().events_binned, 
                Names_Datasets().sets_binned,
            )
            else False
            if self.name in (
                Names_Datasets().images,
                Names_Datasets().sets_unbinned,
            )
            else None
        )

    def _set_name_label(self):

        self.name_label = (
            Names_Labels().binned if self.is_binned
            else Names_Labels().unbinned
        )
        
