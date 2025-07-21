
import torch 

from ...file_handling import save_torch_tensor_as_file
from ..constants import (
    Names_of_Levels, 
    Names_of_Variables,
    Names_of_Labels,
)
from .aggregated_signal import Aggregated_Signal_Dataframe_Handler
from .preprocessing import apply_signal_preprocessing
from .set_creation import (
    make_unbinned_labeled_signal_sets,
    make_binned_labeled_signal_sets,
    make_background_sets,
    shuffle_feature_and_label_tensors
)
from .image_creation import make_image
from .tensor_conversion import torch_tensor_from_pandas


def make_and_save_unbinned_sets_dataset(settings, verbose=True):

    signal_feature_sets, label_sets = make_unbinned_labeled_signal_sets(
        settings=settings,
        list_of_variables_to_standard_scale=Names_of_Variables().list_,
        verbose=verbose
    )

    background_feature_sets = (
        make_background_sets(
            settings=settings,
            num_sets=len(signal_feature_sets),
            verbose=verbose
        ) if settings.set.bkg_fraction is not None
        else torch.tensor([])
    )

    feature_sets = torch.cat([signal_feature_sets, background_feature_sets], dim=1)
    feature_sets, label_sets = shuffle_feature_and_label_tensors(
        feature_tensor=feature_sets, 
        label_tensor=label_sets
    )

    save_torch_tensor_as_file(
        tensor=feature_sets,
        path=settings.features_filepath,
        verbose=verbose
    )
    save_torch_tensor_as_file(
        tensor=label_sets,
        path=settings.labels_filepath,
        verbose=verbose
    )


def make_and_save_binned_sets_dataset(settings, verbose=True):
    
    signal_feature_sets, label_sets, bin_map = make_binned_labeled_signal_sets(
        settings=settings,
        verbose=verbose
    )

    background_feature_sets = make_background_sets(
        settings=settings,
        num_sets=len(signal_feature_sets),
        verbose=verbose
    )

    feature_sets = torch.cat(
        [signal_feature_sets, background_feature_sets],
        dim=1
    )
    feature_sets, label_sets = shuffle_feature_and_label_tensors(
        feature_tensor=feature_sets, 
        label_tensor=label_sets
    )
    
    save_torch_tensor_as_file(
        tensor=feature_sets,
        path=settings.features_filepath,
        verbose=verbose
    )
    save_torch_tensor_as_file(
        tensor=label_sets,
        path=settings.labels_filepath,
        verbose=verbose
    )
    save_torch_tensor_as_file(
        tensor=bin_map,
        path=settings.bin_map_filepath,
        verbose=verbose
    )


def make_and_save_images_dataset(settings, verbose=True):
    
    signal_feature_sets, label_sets = make_unbinned_labeled_signal_sets(
        settings=settings,
        list_of_variables_to_standard_scale=[Names_of_Variables().q_squared,],
        verbose=verbose
    )
    
    background_feature_sets = make_background_sets(
        settings=settings,
        num_sets=len(signal_feature_sets),
        verbose=verbose
    )
    
    feature_sets = torch.cat(
        [signal_feature_sets, background_feature_sets],
        dim=1
    )
    
    list_of_images = [
        make_image(
            feature_set_tensor=set_, 
            num_bins_per_dimension=10
        ).unsqueeze(dim=0) 
        for set_ in feature_sets
    ]
    tensor_of_images = torch.cat(list_of_images)
    tensor_of_images, label_sets = shuffle_feature_and_label_tensors(
        feature_tensor=tensor_of_images, 
        label_tensor=label_sets
    )
    
    save_torch_tensor_as_file(
        tensor=tensor_of_images,
        path=settings.features_filepath,
        verbose=verbose
    )
    save_torch_tensor_as_file(
        tensor=label_sets,
        path=settings.labels_filepath,
        verbose=verbose
    )


def make_and_save_binned_events_dataset(settings, verbose=True):
    
    if settings.common.level == Names_of_Levels().detector_and_background:
        raise NotImplementedError
    
    aggregated_signal_dataframe_handler = Aggregated_Signal_Dataframe_Handler(
        path_to_main_datasets_dir=settings.common.path_to_main_datasets_dir, 
        level=settings.common.level, 
        trial_range=settings.common.raw_signal_trial_range
    )
    aggregated_signal_dataframe = aggregated_signal_dataframe_handler.get_dataframe()
    bin_map = aggregated_signal_dataframe_handler.get_bin_map()
    
    aggregated_signal_dataframe = apply_signal_preprocessing(
        dataframe=aggregated_signal_dataframe, 
        settings=settings, 
        list_of_variables_to_standard_scale=Names_of_Variables().list_
    )

    features = torch_tensor_from_pandas(
        aggregated_signal_dataframe[Names_of_Variables().list_]
    )
    labels = torch_tensor_from_pandas(
        aggregated_signal_dataframe[Names_of_Labels().binned]
    )

    save_torch_tensor_as_file(
        tensor=features, 
        path=settings.features_filepath, 
        verbose=verbose
    )
    save_torch_tensor_as_file(
        tensor=labels,
        path=settings.labels_filepath,
        verbose=verbose
    )
    save_torch_tensor_as_file(
        tensor=bin_map,
        path=settings.bin_map_filepath,
        verbose=verbose
    )
