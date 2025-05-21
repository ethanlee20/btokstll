
from .config import Config_Dataset
from .constants import Names_Datasets
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

        if config.name == Names_Datasets().events_binned:

            self._load_events_binned()

        elif config.name == Names_Datasets().sets_binned:

            self._load_sets_binned()

        elif config.name == Names_Datasets().sets_unbinned:

            self._load_sets_unbinned()

        elif config.name == Names_Datasets().images:

            self._load_images()

        else:

            raise ValueError(
                f"Name not recognized: {config.name}"
            )
        
        print(f"Loaded dataset: {config.name}")

    def _load_events_binned(self):

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

    def _load_sets_binned(self):

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

    def _load_sets_unbinned(self):

        config = self.config

        self.features = load_file_torch_tensor(
            config.path_file_features
        )

        self.labels = load_file_torch_tensor(
            config.path_file_labels
        )

        self.bin_map = None

    def _load_images(self):

        config = self.config

        self.features = load_file_torch_tensor(
            config.path_file_features
        )
        
        self.labels = load_file_torch_tensor(
            config.path_file_labels
        )
        
        self.bin_map = None