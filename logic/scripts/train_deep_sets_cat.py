
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from torch.nn import CrossEntropyLoss

from logic.scripts.library.plotting import setup_mpl_params
from library.modeling.models.deep_sets_cat import Deep_Sets_Cat
from library.data.datasets.bootstrapped_sets_binned import Bootstrapped_Sets_Binned_Dataset
from logic.scripts.library.modeling.training import train_and_eval
from library.modeling.lin_test import plot_linearity
from library.modeling.util import select_device, print_gpu_memory_summary


if __name__ == "__main__":

    run_name = "deep_sets_gen_cat"
    level = "gen"
    model = Deep_Sets_Cat()

    learning_rate = 3e-4
    epochs = 100
    train_batch_size = 32
    eval_batch_size = 32

    device = select_device()

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataset = Bootstrapped_Sets_Binned_Dataset()
    train_dataset.load(level, "train", "../../state/new_physics/data/processed")
    eval_dataset = Bootstrapped_Sets_Binned_Dataset()
    eval_dataset.load(level, "eval", "../../state/new_physics/data/processed")
    lin_eval_dataset = Bootstrapped_Sets_Binned_Dataset() 
    lin_eval_dataset.load(level, "lin_eval", "../../state/new_physics/data/processed")

    models_dir = "../../state/new_physics/models"
    plots_dir = "../../state/new_physics/plots"

    setup_mpl_params()

    train_and_eval(
        run_name, model, train_dataset, eval_dataset,
        loss_fn, optimizer, device,
        models_dir, plots_dir,
        epochs, train_batch_size, eval_batch_size
    )

    print_gpu_memory_summary()

    # plot_linearity(run_name, model, lin_eval_dataset, device, plots_dir)
