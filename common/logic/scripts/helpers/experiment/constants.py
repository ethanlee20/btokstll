
from pathlib import Path


class Paths_to_Directories:

    path_to_main_datasets_dir = Path("../../state/new_physics/data/processed")
    path_to_raw_signal_dir = Path("../../state/new_physics/data/raw/signal")
    path_to_raw_bkg_dir = Path("../../state/new_physics/data/raw/bkg")
    path_to_main_models_dir = Path("../../state/new_physics/models")
    path_to_plots_dir = Path("../../state/new_physics/plots")


class Names_of_Result_Table_Columns:

    mse = "MSE"
    mae = "MAE"
    np_std = "Std. at NP"
    np_mean = "Mean at NP"
    np_bias = "Bias at NP"

    tuple_ = (mse, mae, np_std, np_mean, np_bias)


delta_C9_value_new_physics = -0.82
delta_C9_value_standard_model = 0