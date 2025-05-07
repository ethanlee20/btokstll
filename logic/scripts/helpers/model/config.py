
import pathlib

from.util import select_device
from ..data.dset.config import Dataset_Config


class Model_Config:
    """
    Model configuration.
    """

    def __init__(
        self,
        name,
        path_dir_models_main,
        dset_config:Dataset_Config,
        loss_fn=None,
        optimizer=None,
        lr_scheduler=None,
    ):
        """
        Initialize.

        path_dir_models_main : str
            Path to the main models directory.
        dset_config : Dataset_Config
            Config for either 
            train or eval dataset.
        """        

        self.name = name
        self.path_dir_models_main = (
            pathlib.Path(
                path_dir_models_main
            )
        )
        self.dset_config = dset_config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = select_device()

        self._load_constants()
        self._check_inputs()
        self._make_path_dir()
        self._make_path_file_final()
        self._make_path_file_loss_table()

    def make_path_file_checkpoint(self, epoch:int):
        """
        Make the path of a checkpoint model file.
        """

        name = f"epoch_{epoch}.pt"
        path = self.path_dir.joinpath(name)
        return path

    def _make_path_file_final(self):
        """
        Make the path of the final model file.
        """

        name = "final.pt"
        self.path_file_final = (
            self.path_dir.joinpath(
                name
            )
        )

    def _make_path_file_loss_table(self):
        """
        Make the path of the loss table.
        """

        name = "loss_table.pkl"
        self.path_file_loss_table = (
            self.path_dir.joinpath(
                name
            )
        )

    def _make_path_dir(self):
        """
        Make the path of the model's directory.
        """

        self.path_dir_model_type = (
            self.path_dir_models_main.joinpath(
                self.name
            )
        )

        name_dir = (
            f"{self.dset_config.num_events_per_set}_"
            f"{self.dset_config.level}_"
            f"q2v_{self.dset_config.q_squared_veto}"
        )

        self.path_dir = (
            self.path_dir_model_type.joinpath(
                name_dir
            )
        )
    
    def _load_constants(self):
        """
        Load constants.
        """

        self.name_model_deep_sets = "deep_sets"
        self.name_model_cnn = "cnn"
        self.name_model_ebe = "ebe"

        self.names_models = (
            self.name_model_deep_sets,
            self.name_model_cnn,
            self.name_model_ebe,
        )
    
    def _check_inputs(self,):
        """
        Check that inputs make sense.
        """
        
        if self.name not in self.names_models:
            raise ValueError(
                f"Name not recognized: {self.name}"
            )
        
        if not self.path_dir_models_main.is_dir():
            raise ValueError(
                "Main models directory "
                "is not a directory."
            )
