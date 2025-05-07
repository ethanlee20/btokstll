
import torch

from .config import Config
from .generator import Dataset_Generator
from .loader import Dataset_Loader


class Custom_Dataset(torch.utils.data.Dataset):
    """
    Custom dataset.
    """
    
    def __init__(
        self, 
        config:Config,
    ):
        """
        Initialize.
        """
    
        self.config = config

    def generate(self):
        """
        Generate.
        """

        generator = Dataset_Generator(
            self.config
        )
        generator.generate()


    def load(self):
        """
        Load.
        """

        loader = Dataset_Loader(
            self.config
        )
        loader.load()
        self.features = loader.features
        self.labels = loader.labels
        self.bin_map = loader.bin_map

    def unload(self):
        """
        Unload data from memory.
        """

        del self.features
        del self.labels
        del self.bin_map

        print("Unloaded dataset.")

    def __len__(self):
        """
        Get the length of the dataset.
        """

        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Get a (feature, label) pair.

        Parameters
        ----------
        index : int
            The index of the pair within the dataset.

        Returns
        -------
        x : torch.Tensor
            The features.
        y : torch.Tensor
            The label.
        """

        x = self.features[index]
        y = self.labels[index]
        return x, y


