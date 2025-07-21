
from pathlib import Path

from .constants import Names_of_Models


class Model_Training_Settings:

    def __init__(
        self,
        training_dataset_settings,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints
    ):
        self.training_dataset_settings = training_dataset_settings
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.learning_rate_scheduler_reduction_factor = learning_rate_scheduler_reduction_factor
        self.size_of_training_batch = size_of_training_batch
        self.size_of_evaluation_batch = size_of_evaluation_batch
        self.number_of_epochs = number_of_epochs
        self.number_of_epochs_between_checkpoints = number_of_epochs_between_checkpoints


def make_path_to_parent_model_dir(path_to_main_models_dir, model_name):
    
    dir_name = f"{model_name}"
    path_to_parent_model_dir = Path(path_to_main_models_dir).joinpath(dir_name)
    return path_to_parent_model_dir


def make_path_to_event_based_model_dir(path_to_main_models_dir, model_name, level, q_squared_veto):
   
    path_to_parent_model_dir = make_path_to_parent_model_dir(
        path_to_main_models_dir=path_to_main_models_dir, 
        model_name=model_name
    )
    dir_name = f"{level}_q2v_{q_squared_veto}"
    path_to_model_dir = path_to_parent_model_dir.joinpath(dir_name)
    return path_to_model_dir


def make_path_to_set_based_model_dir(path_to_main_models_dir, model_name, num_events_per_set, level, q_squared_veto):
    
    path_to_parent_model_dir = make_path_to_parent_model_dir(
        path_to_main_models_dir=path_to_main_models_dir,
        model_name=model_name
    )
    dir_name = f"{num_events_per_set}_{level}_q2v_{q_squared_veto}"
    path_to_model_dir = path_to_parent_model_dir.joinpath(dir_name)
    return path_to_model_dir


class Model_Paths:

    def __init__(
        self,
        model_name,
        path_to_main_models_dir,
        training_dataset_settings,
    ):  
        self.path_to_main_models_dir = path_to_main_models_dir
        self.path_to_model_dir = (
            make_path_to_set_based_model_dir(
                path_to_main_models_dir=path_to_main_models_dir,
                model_name=model_name,
                num_events_per_set=training_dataset_settings.set.num_events_per_set,
                level=training_dataset_settings.common.level,
                q_squared_veto=training_dataset_settings.preprocessing.q_squared_veto
            ) if model_name in Names_of_Models().set_based
            else make_path_to_event_based_model_dir(
                path_to_main_models_dir=path_to_main_models_dir,
                model_name=model_name,
                level=training_dataset_settings.common.level,
                q_squared_veto=training_dataset_settings.preprocessing.q_squared_veto
            ) if model_name in Names_of_Models().event_based
            else None
        )
        if self.path_to_model_dir is None: raise ValueError

    def make_path_to_final_model_file(self):
        name_of_file = f"final.pt"
        path = self.path_to_model_dir.joinpath(name_of_file)
        return path

    def make_path_to_checkpoint_model_file(self, epoch):
        name_of_file = f"epoch_{epoch}.pt"
        path = self.path_to_model_dir.joinpath(name_of_file)
        return path
    
    def make_path_to_loss_table_file(self):
        name_of_file = f"loss_table.pkl"
        path = self.path_to_model_dir.joinpath(name_of_file)
        return path
    
    def make_path_to_loss_curves_plot(self):
        name_of_file = "loss_curves.png"
        path = self.path_to_model_dir.joinpath(name_of_file)
        return path


class Model_Settings:

    def __init__(
        self,
        name,
        path_to_main_models_dir,
        training_dataset_settings,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints
    ):
        self.name = name
        self.paths = Model_Paths(
            model_name=name,
            path_to_main_models_dir=path_to_main_models_dir,
            training_dataset_settings=training_dataset_settings
        )
        self.training = Model_Training_Settings(
            training_dataset_settings=training_dataset_settings,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints
        )

        

