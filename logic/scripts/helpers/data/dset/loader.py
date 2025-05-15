
from .config import Config_Dataset
from ..file_hand import load_file_torch_tensor


class Dataset_Loader:
    """
    Loads a dataset.
    """
    def __init__(
        self, 
        config:Config_Dataset
    ):
        
        self.config = config
    
    def load(self):

        """
        Load specified dataset.
        """

        config = self.config

        if config.name == config.name_dset_binned_signal:

            self._load_binned_signal()

        elif config.name == config.name_dset_sets_binned_signal:

            self._load_sets_binned_signal()

        elif config.name == config.name_dset_sets_unbinned_signal:

            self._load_sets_unbinned_signal()

        elif config.name == config.name_dset_images_signal:

            self._load_images_signal()

        else:

            raise ValueError(
                f"Name not recognized: {config.name}"
            )
        
        print(f"Loaded dataset: {config.name}")

    def _load_binned_signal(self):

        config = self.config

        self.features = load_file_torch_tensor(
            config.path_file_features
        )

        self.labels = load_file_torch_tensor(
            config.path_file_labels
        )

        self.bin_map = load_file_torch_tensor(
            config.path_file_bin_map
        )

    def _load_sets_binned_signal(self):

        config = self.config

        self.features = load_file_torch_tensor(
            config.path_file_features
        )

        self.labels = load_file_torch_tensor(
            config.path_file_labels
        )

        self.bin_map = load_file_torch_tensor(
            config.path_file_bin_map
        )

    def _load_sets_unbinned_signal(self):

        config = self.config

        self.features = load_file_torch_tensor(
            config.path_file_features
        )

        self.labels = load_file_torch_tensor(
            config.path_file_labels
        )

        self.bin_map = None

    def _load_images_signal(self):

        config = self.config

        self.features = load_file_torch_tensor(
            config.path_file_features
        )
        
        self.labels = load_file_torch_tensor(
            config.path_file_labels
        )
        
        self.bin_map = None