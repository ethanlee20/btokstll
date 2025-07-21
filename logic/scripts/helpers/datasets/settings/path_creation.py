
from pathlib import Path

from ..constants import Names_of_Kinds_of_Dataset_Files


def make_path_to_dataset_dir(
    name_dataset, 
    level, 
    q_squared_veto, 
    path_to_main_datasets_dir,
):
    
    def _make_name_of_dataset_dir(dataset_name, level, q_squared_veto):
        return f"{dataset_name}_{level}_q2v_{q_squared_veto}"
    
    name = _make_name_of_dataset_dir(
        dataset_name=name_dataset,
        level=level,
        q_squared_veto=q_squared_veto
    )
    path = (
        Path(path_to_main_datasets_dir)
        .joinpath(name)
    )
    return path


def _make_path_of_set_based_dataset_file(
    kind,
    split,
    num_events_per_set,
    is_sensitivity_study,
    path_of_dataset_dir
):
    
    def _make_name_of_set_based_dataset_file(
        kind:str, 
        split:str,
        num_events_per_set:int,
        is_sensitivity_study:bool
    ):
        if kind not in Names_of_Kinds_of_Dataset_Files().tuple_:
            raise ValueError  
        sensitivity_token = "sens_" if is_sensitivity_study else ""
        name = f"{num_events_per_set}_{split}_{sensitivity_token}{kind}.pt"
        return name
    
    name = _make_name_of_set_based_dataset_file(
        kind=kind,
        split=split,
        num_events_per_set=num_events_per_set,
        is_sensitivity_study=is_sensitivity_study
    )
    path = Path(path_of_dataset_dir).joinpath(name)
    return path


def make_path_to_set_based_features_file(
    split,
    num_events_per_set,
    is_sensitivity_study,
    path_to_dataset_dir
):

    path = _make_path_of_set_based_dataset_file(
        kind=Names_of_Kinds_of_Dataset_Files().features,
        split=split,
        num_events_per_set=num_events_per_set,
        is_sensitivity_study=is_sensitivity_study,
        path_of_dataset_dir=path_to_dataset_dir
    )
    return path


def make_path_to_set_based_labels_file(
    split,
    num_events_per_set,
    is_sensitivity_study,
    path_to_dataset_dir
):

    path = _make_path_of_set_based_dataset_file(
        kind=Names_of_Kinds_of_Dataset_Files().labels,
        split=split,
        num_events_per_set=num_events_per_set,
        is_sensitivity_study=is_sensitivity_study,
        path_of_dataset_dir=path_to_dataset_dir
    )
    return path


def make_path_to_set_based_bin_map_file(
    split,
    num_events_per_set,
    is_sensitivity_study,
    path_to_dataset_dir
):

    path = _make_path_of_set_based_dataset_file(
        kind=Names_of_Kinds_of_Dataset_Files().bin_map,
        split=split,
        num_events_per_set=num_events_per_set,
        is_sensitivity_study=is_sensitivity_study,
        path_of_dataset_dir=path_to_dataset_dir
    )
    return path


def _make_path_of_event_based_dataset_file(kind, split, path_of_dataset_dir):

    def _make_name_of_event_based_dataset_file(kind, split):    
        if kind not in Names_of_Kinds_of_Dataset_Files().tuple_:
            raise ValueError
        return f"{split}_{kind}.pt"
    
    name = _make_name_of_event_based_dataset_file(kind=kind, split=split)
    path = Path(path_of_dataset_dir).joinpath(name)
    return path


def make_path_to_event_based_features_file(split, path_to_dataset_dir):
    
    path = _make_path_of_event_based_dataset_file(
        kind=Names_of_Kinds_of_Dataset_Files().features,
        split=split,
        path_of_dataset_dir=path_to_dataset_dir
    )
    return path


def make_path_to_event_based_labels_file(split, path_to_dataset_dir):
    
    path = _make_path_of_event_based_dataset_file(
        kind=Names_of_Kinds_of_Dataset_Files().labels,
        split=split,
        path_of_dataset_dir=path_to_dataset_dir
    )
    return path


def make_path_to_event_based_bin_map_file(split, path_to_dataset_dir):
    
    path = _make_path_of_event_based_dataset_file(
        kind=Names_of_Kinds_of_Dataset_Files().bin_map,
        split=split,
        path_of_dataset_dir=path_to_dataset_dir
    )
    return path



