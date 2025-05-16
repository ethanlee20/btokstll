
import pathlib

from .util import select_device
from .constants import Names_Models
from ..data.dset.config import Config_Dataset


class Config_Model:

    """
    Model configuration.
    """

    def __init__(
        self,
        name:str,
        path_dir_models_main:str|pathlib.Path,
        config_dset_train:Config_Dataset,
        loss_fn=None,
        optimizer=None,
        lr_scheduler=None,
        size_batch_train=None,
        size_batch_eval=None,
        num_epochs=None,
        num_epochs_checkpoint=None,
    ):
        """
        Initialize.
        """
        
        self.name = name

        self.path_dir_models_main = (
            pathlib.Path(
                path_dir_models_main
            )
        )

        self.config_dset_train = config_dset_train

        self.fn_loss = loss_fn

        self.optimizer = optimizer

        self.scheduler_lr = lr_scheduler

        self.size_batch_train = size_batch_train
        self.size_batch_eval = size_batch_eval

        self.num_epochs = num_epochs
        self.num_epochs_checkpoint = num_epochs_checkpoint

        self._check_inputs()

        self._set_path_dir()

        self._set_path_file_final()
        self._set_path_file_loss_table()

    def make_path_file_checkpoint(self, epoch:int):

        """
        Make the path of a checkpoint model file.
        """

        name = f"epoch_{epoch}.pt"

        path = self.path_dir.joinpath(name)

        return path

    def _set_path_file_final(self):

        """
        Make the path of the final model file.
        """

        name = "final.pt"

        self.path_file_final = (
            self.path_dir.joinpath(
                name
            )
        )

    def _set_path_file_loss_table(self):

        """
        Make the path of the loss table.
        """

        name = "loss_table.pkl"

        self.path_file_loss_table = (
            self.path_dir.joinpath(
                name
            )
        )

    def _set_path_dir(self):

        """
        Make the path of the model's directory.
        """

        self.path_dir_model_type = (
            self.path_dir_models_main.joinpath(
                self.name
            )
        )

        name_dir = (
            f"{self.config_dset_train.level}_"
            f"q2v_{self.config_dset_train.q_squared_veto}"
        )

        if self.config_dset_train.num_events_per_set:
            name_dir = (
                f"{self.config_dset_train.num_events_per_set}_"
                + name_dir
            )

        self.path_dir = (
            self.path_dir_model_type.joinpath(
                name_dir
            )
        )
    
    def _check_inputs(self,):

        """
        Check that inputs make sense.
        """

        names_models = Names_Models()

        if self.name not in (
            names_models.tuple_
        ):
            
            raise ValueError(
                f"Name not recognized: {self.name}"
            )
        
        if not self.path_dir_models_main.is_dir():
            
            raise ValueError(
                "Main models directory "
                "is not a directory."
            )



