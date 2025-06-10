
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
        Load dataset specified in config. 
        """

        if self.config.name not in Names_Datasets().tuple_:
            raise ValueError(
                f"Name not recognized: {self.config.name}"
            )
        
        self.features = load_file_torch_tensor(
            self.config.path_file_features
        )
        self.labels = load_file_torch_tensor(
            self.config.path_file_labels
        )
        self.bin_map = (
            load_file_torch_tensor(
                self.config.path_file_bin_map
            ) if self.config.is_binned
            else None
        )
        
        print(f"Loaded dataset: {self.config.name}")