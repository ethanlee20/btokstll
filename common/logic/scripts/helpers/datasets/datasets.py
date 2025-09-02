
import torch

from .make_and_save.make_and_save import (
    make_and_save_unbinned_sets_dataset,
    make_and_save_binned_sets_dataset,
    make_and_save_images_dataset,
    make_and_save_binned_events_dataset
)
from ..file_handling import load_torch_tensor_from_file


def print_unloaded_datasets_message():
    print("Unloaded datasets.")


class Unbinned_Sets_Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        settings,
        verbose=True
    ):
        self.settings = settings
        self.verbose = verbose

    def make_and_save(self):
        self.settings.common.path_to_dataset_dir.mkdir(parents=True, exist_ok=True)
        make_and_save_unbinned_sets_dataset(
            settings=self.settings,
            verbose=self.verbose
        )

    def load(self):
        self.features = load_torch_tensor_from_file(
            path=self.settings.features_filepath,
            verbose=self.verbose
        )
        self.labels = load_torch_tensor_from_file(
            path=self.settings.labels_filepath,
            verbose=self.verbose
        )

    def unload(self):
        del self.features
        del self.labels
        if self.verbose:
            print_unloaded_datasets_message()

    def __len__(self):
        return len(self.labels)


class Binned_Sets_Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        settings,
        verbose=True
    ):
        self.settings = settings
        self.verbose = verbose

    def make_and_save(self):
        self.settings.common.path_to_dataset_dir.mkdir(parents=True, exist_ok=True)
        make_and_save_binned_sets_dataset(
            settings=self.settings,
            verbose=self.verbose
        )

    def load(self):
        self.features = load_torch_tensor_from_file(
            path=self.settings.features_filepath,
            verbose=self.verbose
        )
        self.labels = load_torch_tensor_from_file(
            path=self.settings.labels_filepath,
            verbose=self.verbose
        )
        self.bin_map = load_torch_tensor_from_file(
            path=self.settings.bin_map_filepath, 
            verbose=self.verbose
        )

    def unload(self):
        del self.features
        del self.labels
        del self.bin_map
        if self.verbose:
            print_unloaded_datasets_message()

    def __len__(self):
        return len(self.labels)
    

class Images_Dataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        settings,
        verbose=True
    ):  
        self.settings = settings
        self.verbose = verbose

    def make_and_save(self):
        self.settings.common.path_to_dataset_dir.mkdir(parents=True, exist_ok=True)
        make_and_save_images_dataset(
            settings=self.settings,
            verbose=self.verbose
        )

    def load(self):
        self.features = load_torch_tensor_from_file(
            path=self.settings.features_filepath,
            verbose=self.verbose
        )
        self.labels = load_torch_tensor_from_file(
            path=self.settings.labels_filepath,
            verbose=self.verbose
        )

    def unload(self):
        del self.features
        del self.labels
        if self.verbose:
            print_unloaded_datasets_message()

    def __len__(self):
        return len(self.labels)
    

class Binned_Events_Dataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        settings,
        verbose=True
    ):
        self.settings = settings
        self.verbose = verbose

    def make_and_save(self):
        self.settings.common.path_to_dataset_dir.mkdir(parents=True, exist_ok=True)
        make_and_save_binned_events_dataset(
            settings=self.settings,
            verbose=self.verbose
        )

    def load(self):
        self.features = load_torch_tensor_from_file(
            path=self.settings.features_filepath,
            verbose=self.verbose
        )
        self.labels = load_torch_tensor_from_file(
            path=self.settings.labels_filepath,
            verbose=self.verbose
        )
        self.bin_map = load_torch_tensor_from_file(
            path=self.settings.bin_map_filepath,
            verbose=self.verbose
        )

    def unload(self):
        del self.features
        del self.labels
        del self.bin_map
        if self.verbose:
            print_unloaded_datasets_message()

    def __len__(self):
        return len(self.labels)
