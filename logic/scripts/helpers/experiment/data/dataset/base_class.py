
import torch


class Custom_Dataset(torch.utils.data.Dataset):
    
    """
    Custom dataset base class.
    
    All project datasets follow
    this template.
    """
    
    def __init__(
        self, 
        config:Dataset_Config,
        generator:Dataset_Generator,
    ):
        
        """
        Initialize.

        Parameters
        ----------
        config : Dataset_Config
            Configuration.
        
        Side Effects
        ------------
        - Create a subdirectory for the dataset
        in the directory specified by save_dir
        (if it doesn't already exist).
        """
    
        self.config = config

        self.config.sub_dir_path.mkdir(exist_ok=True)

    def load(self):
        """
        Overwrite this with a function that loads necessary files
        (at least set self.features and self.labels).
        """
        pass

    def unload(self):
        """
        Overwrite this with a function that unloads the
        loaded data from memory (self.features and self.labels).
        """
        pass

    def generate(self):
        """
        Overwrite this with a function that saves necessary files
        (at least features and labels).
        """
        pass



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

