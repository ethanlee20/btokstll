
import torch

from .config import Config_Dataset
from .generator import Dataset_Generator
from .loader import Dataset_Loader


class Custom_Dataset(torch.utils.data.Dataset):

    """
    Custom dataset.
    """
    
    def __init__(
        self, 
        config:Config_Dataset,
    ):
        self.config = config

    def generate(self):
        
        """
        Create and save the dataset.
        """

        generator = Dataset_Generator(
            self.config
        )
        
        generator.generate()


    def load(self):

        """
        Load the dataset from disk.
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
        """

        x = self.features[index]
        y = self.labels[index]
        return x, y


