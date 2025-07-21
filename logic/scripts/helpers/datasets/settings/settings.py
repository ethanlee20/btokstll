
from ..constants import Names_of_Datasets, Names_of_Levels
from .data_handling import (
    get_raw_signal_trial_range, 
    calc_num_signal_bkg_per_set
)
from .path_creation import (
    make_path_to_dataset_dir,
    make_path_to_set_based_features_file,
    make_path_to_set_based_labels_file,
    make_path_to_set_based_bin_map_file,
    make_path_to_event_based_features_file,
    make_path_to_event_based_labels_file,
    make_path_to_event_based_bin_map_file
)


class Common_Settings:

    def __init__(
        self,
        name,
        level,
        split,
        q_squared_veto,
        std_scale,
        shuffle,
        path_to_main_datasets_dir,
        path_to_raw_signal_dir,
        path_to_raw_bkg_dir,
        label_subset=None,
    ):   
        
        self.name = name
        self.level = level
        self.split = split
        self.path_to_main_datasets_dir = path_to_main_datasets_dir
        self.path_to_raw_signal_dir = path_to_raw_signal_dir
        self.path_to_raw_bkg_dir = path_to_raw_bkg_dir
        self.preprocessing = self.Preprocessing_Settings(
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            label_subset=label_subset
        )
        self.raw_signal_trial_range = get_raw_signal_trial_range(split)
        self.path_to_dataset_dir = make_path_to_dataset_dir(
            name_dataset=name,
            level=level,
            q_squared_veto=self.preprocessing.q_squared_veto,
            path_to_main_datasets_dir=path_to_main_datasets_dir
        )

    class Preprocessing_Settings:

        def __init__(
            self,
            q_squared_veto,
            std_scale,
            shuffle,
            label_subset=None,
        ):
            
            self.q_squared_veto = q_squared_veto
            self.std_scale = std_scale
            self.shuffle = shuffle
            self.label_subset = label_subset


class Set_Settings:

    def __init__(
        self,
        num_events_per_set,
        num_sets_per_label,
        is_sensitivity_study,
        bkg_fraction=None,
        bkg_charge_fraction=None, # fraction of background that comes from charge
    ):
        
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.bkg_fraction = bkg_fraction
        self.bkg_charge_fraction = bkg_charge_fraction 
        self.is_sensitivity_study = is_sensitivity_study
        self.num_events_per_set_signal, self.num_events_per_set_bkg = (
            calc_num_signal_bkg_per_set(
                num_events_per_set=num_events_per_set,
                bkg_fraction=bkg_fraction
            )
        )


class Unbinned_Sets_Dataset_Settings:

    def __init__(
        self,
        level,
        split,
        num_events_per_set,
        num_sets_per_label,
        is_sensitivity_study,
        q_squared_veto,
        std_scale,
        shuffle,
        path_to_main_datasets_dir,
        path_to_raw_signal_dir,
        path_to_raw_bkg_dir,
        bkg_fraction=None,
        bkg_charge_fraction=None,
        label_subset=None
    ):
        
        self.common = Common_Settings(
            name=Names_of_Datasets().sets_unbinned,
            level=level,
            split=split,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            path_to_main_datasets_dir=path_to_main_datasets_dir,
            path_to_raw_signal_dir=path_to_raw_signal_dir,
            path_to_raw_bkg_dir=path_to_raw_bkg_dir,
            label_subset=label_subset
        )
        self.set = Set_Settings(
            num_events_per_set=num_events_per_set,
            num_sets_per_label=num_sets_per_label,
            is_sensitivity_study=is_sensitivity_study,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction
        )

        self.features_filepath = make_path_to_set_based_features_file(
            split=split,
            num_events_per_set=num_events_per_set,
            is_sensitivity_study=is_sensitivity_study,
            path_to_dataset_dir=self.common.path_to_dataset_dir
        )
        self.labels_filepath = make_path_to_set_based_labels_file(
            split=split,
            num_events_per_set=num_events_per_set,
            is_sensitivity_study=is_sensitivity_study,
            path_to_dataset_dir=self.common.path_to_dataset_dir
        )


class Binned_Sets_Dataset_Settings:

    def __init__(
        self,
        level,
        split,
        num_events_per_set,
        num_sets_per_label,
        is_sensitivity_study,
        q_squared_veto,
        std_scale,
        shuffle,
        path_to_main_datasets_dir,
        path_to_raw_signal_dir,
        path_to_raw_bkg_dir,
        bkg_fraction=None,
        bkg_charge_fraction=None,
        label_subset=None
    ):
        
        self.common = Common_Settings(
            name=Names_of_Datasets().sets_binned,
            level=level,
            split=split,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            path_to_main_datasets_dir=path_to_main_datasets_dir,
            path_to_raw_signal_dir=path_to_raw_signal_dir,
            path_to_raw_bkg_dir=path_to_raw_bkg_dir,
            label_subset=label_subset
        )
        self.set = Set_Settings(
            num_events_per_set=num_events_per_set,
            num_sets_per_label=num_sets_per_label,
            is_sensitivity_study=is_sensitivity_study,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction
        )

        self.features_filepath = make_path_to_set_based_features_file(
            split=split,
            num_events_per_set=num_events_per_set,
            is_sensitivity_study=is_sensitivity_study,
            path_to_dataset_dir=self.common.path_to_dataset_dir
        )
        self.labels_filepath = make_path_to_set_based_labels_file(
            split=split,
            num_events_per_set=num_events_per_set,
            is_sensitivity_study=is_sensitivity_study,
            path_to_dataset_dir=self.common.path_to_dataset_dir
        )
        self.bin_map_filepath = make_path_to_set_based_bin_map_file(
            split=split,
            num_events_per_set=num_events_per_set,
            is_sensitivity_study=is_sensitivity_study,
            path_to_dataset_dir=self.common.path_to_dataset_dir
        )


class Images_Dataset_Settings:

    def __init__(
        self,
        level,
        split,
        num_events_per_set,
        num_sets_per_label,
        num_bins_per_dimension,
        is_sensitivity_study,
        q_squared_veto,
        std_scale,
        shuffle,
        path_to_main_datasets_dir,
        path_to_raw_signal_dir,
        path_to_raw_bkg_dir,
        bkg_fraction=None,
        bkg_charge_fraction=None,
        label_subset=None
    ):
        
        self.common = Common_Settings(
            name=Names_of_Datasets().images,
            level=level,
            split=split,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            path_to_main_datasets_dir=path_to_main_datasets_dir,
            path_to_raw_signal_dir=path_to_raw_signal_dir,
            path_to_raw_bkg_dir=path_to_raw_bkg_dir,
            label_subset=label_subset
        )
        self.set = Set_Settings(
            num_events_per_set=num_events_per_set,
            num_sets_per_label=num_sets_per_label,
            is_sensitivity_study=is_sensitivity_study,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction
        )
        self.num_bins_per_dimension = num_bins_per_dimension
        self.features_filepath = make_path_to_set_based_features_file(
            split=split,
            num_events_per_set=num_events_per_set,
            is_sensitivity_study=is_sensitivity_study,
            path_to_dataset_dir=self.common.path_to_dataset_dir
        )
        self.labels_filepath = make_path_to_set_based_labels_file(
            split=split,
            num_events_per_set=num_events_per_set,
            is_sensitivity_study=is_sensitivity_study,
            path_to_dataset_dir=self.common.path_to_dataset_dir
        )


class Binned_Events_Dataset_Settings:

    def __init__(
        self,
        level,
        split, 
        q_squared_veto,
        std_scale,
        shuffle,
        path_to_main_datasets_dir,
        path_to_raw_signal_dir,
        path_to_raw_bkg_dir,
        label_subset=None,
    ):  
        
        if level == Names_of_Levels().detector_and_background:
            raise NotImplementedError
        
        self.basic = Common_Settings(
            name=Names_of_Datasets().events_binned,
            level=level,
            split=split,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            path_to_main_datasets_dir=path_to_main_datasets_dir,
            path_to_raw_signal_dir=path_to_raw_signal_dir,
            path_to_raw_bkg_dir=path_to_raw_bkg_dir,
            label_subset=label_subset
        )

        self.features_filepath = make_path_to_event_based_features_file(
            split=split,
            path_to_dataset_dir=self.basic.path_to_dataset_dir
        )
        self.labels_filepath = make_path_to_event_based_labels_file(
            split=split,
            path_to_dataset_dir=self.basic.path_to_dataset_dir
        )
        self.bin_map_filepath = make_path_to_event_based_bin_map_file(
            split=split,
            path_to_dataset_dir=self.basic.path_to_dataset_dir
        )