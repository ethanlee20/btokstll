import torch

from helpers.model.util import select_device
from helpers.experiment.experiment import Experiment
from helpers.experiment.configs import (
    Config_Experiment_Images,
    Config_Experiment_Deep_Sets,
    Config_Experiment_Event_by_Event
)
from helpers.data.dset.config import Config_Dataset
from helpers.data.dset.constants import (
    Names_Datasets,
    Names_Levels,
    Names_q_Squared_Vetos,
    Names_Splits,
    Names_Variables,
    Nums_Events_Per_Set
)
from helpers.model.config import Config_Model
from helpers.model.constants import Names_Models
from helpers.plot.util import setup_high_quality_mpl_params


setup_high_quality_mpl_params()

path_dir_plots = "../../state/new_physics/plots"


device = select_device()


experiment = Experiment(
    path_dir_plots=path_dir_plots,
    device=device,
)

config_experiment_ebe = Config_Experiment_Event_by_Event()

for level in (Names_Levels().detector,):

    experiment.train(
        config_model=config_experiment_ebe.get_config_model(
            level=level, 
        ),
        config_dset_eval=config_experiment_ebe.get_config_dset(
            level=level, 
            split=Names_Splits().eval_,
        ),
        generate_dsets=True,
    )