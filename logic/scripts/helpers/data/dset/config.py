
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
        split:str,
        q_squared_veto:str,
        balanced_classes:bool, 
        std_scale:bool,
        shuffle:bool,
        path_dir_dsets_main:str|pathlib.Path,
        path_dir_raw_signal:str|pathlib.Path,
        num_events_per_set:int=None,
        num_sets_per_label:int=None,
        num_bins_image:int=None,
        frac_bkg:float=None,
        path_dir_raw_bkg:str|pathlib.Path=None,
        label_subset:list[float]|str=None, # original labels (not bin values)
        is_sensitivity_study:bool=False,
    ):
        
        self.name = name
        self.level = level
        self.split = split

        self.q_squared_veto = q_squared_veto
        self.balanced_classes = balanced_classes
        self.std_scale = std_scale
        self.shuffle = shuffle
        
        self.path_dir_dsets_main = pathlib.Path(
            path_dir_dsets_main
        )
        self.path_dir_raw_signal = pathlib.Path(
            path_dir_raw_signal
        )
        
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.num_bins_image = num_bins_image
        
        self.frac_bkg = frac_bkg
        if path_dir_raw_bkg:
            self.path_dir_raw_bkg = pathlib.Path(
                path_dir_raw_bkg
            )
        
        self.label_subset = label_subset
        self.is_sensitivity_study = is_sensitivity_study
        
        self._check_inputs()
        
        self._set_range_trials_raw_signal()
        
        self._set_path_dir()
        self._set_paths_files()

        self._set_is_binned()
        self._set_name_label()

        if self.name in Names_Datasets().set_based:
            self._set_num_signal_bkg()

    def _check_inputs(self):

        def check_required():

            if self.name not in Names_Datasets().tuple_:
                raise ValueError
            if self.level not in Names_Levels().tuple_:
                raise ValueError
            if self.split not in Names_Splits().tuple_:
                raise ValueError
            if self.q_squared_veto not in Names_q_Squared_Vetos().tuple_:
                raise ValueError
            if type(self.balanced_classes) is not bool:
                raise ValueError
            if type(self.std_scale) is not bool:
                raise ValueError
            if type(self.shuffle) is not bool:
                raise ValueError
            if not self.path_dir_dsets_main.is_dir():
                raise ValueError
            if not self.path_dir_raw_signal.is_dir():
                raise ValueError
            
        def check_set():

            if self.num_events_per_set is None:
                raise ValueError
            if self.num_sets_per_label is None:
                raise ValueError
            if (
                (self.name == Names_Datasets().images) 
                and (type(self.num_bins_image) is not int)
            ):
                raise ValueError

        def check_bkg():

            if self.path_dir_raw_bkg is None:
                raise ValueError
            if not self.path_dir_raw_bkg.is_dir():
                raise ValueError
            if self.frac_bkg:
                if (self.frac_bkg > 1) or (self.frac_bkg < 0):
                    raise ValueError
        
        def check_misc():

            if self.label_subset:
                if type(self.label_subset) not in (list, str):
                    raise ValueError
            if type(self.is_sensitivity_study) is not bool: 
                raise ValueError

        check_required()

        if self.name in Names_Datasets().set_based: 
            check_set()

        if self.level == Names_Levels().detector_and_background: 
            check_bkg()

        check_misc()

    def _set_path_dir(self):

        """
        Create the dataset's directory path.
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
            raise ValueError
        
        sensitivity_label = (
            None if not self.is_sensitivity_study 
            else "sens"
        )

        name_parts = [
            str(i) for i in
            [
                self.num_events_per_set, 
                self.split, 
                sensitivity_label, 
                kind,
            ]
            if i is not None
        ]

        name_base = "_".join(name_parts)
        extension = ".pt"
        name = name_base + extension
        path = self.path_dir.joinpath(name)

        return path

    def _set_paths_files(self):

        """
        Set dataset file paths.
        """

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
        Set the raw signal trial range 
        corresponding to the data split.
        """
        
        self.range_trials_raw_signal = (
            Trials_Splits().train if (
                self.split == Names_Splits().train
            )
            else Trials_Splits().eval_ if (
                self.split == Names_Splits().eval_
            )
            else None
        )

    def _set_num_signal_bkg(self):

        if (
            (self.frac_bkg is None) 
            and (self.level == Names_Levels().detector_and_background)
        ):
            raise ValueError
        if (
            (self.frac_bkg is not None)
            and (self.level != Names_Levels().detector_and_background)
        ):
            raise ValueError
        
        self.num_events_per_set_bkg = (
            int(
                self.num_events_per_set 
                * self.frac_bkg
            ) if self.frac_bkg is not None
            else 0
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
        
