
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from library.utilities.plotting import setup_mpl_params
from library.modeling.models.single_event_nn import Single_Event_NN
from library.data.datasets.aggregated_signal_binned import Aggregated_Signal_Binned_Dataset
from library.modeling.train import train_and_eval
from library.modeling.lin_test import plot_linearity
from library.modeling.util import select_device, print_gpu_memory_summary


if __name__ == "__main__":

    run_name = "single_event_nn_gen"
    level = "gen"
    model = Single_Event_NN()

    learning_rate = 4e-3
    epochs = 100
    train_batch_size = 128
    eval_batch_size = 128

    device = select_device()

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataset = Aggregated_Signal_Binned_Dataset()
    train_dataset.load(level, "train", "../../state/new_physics/data/processed")
    eval_dataset = Aggregated_Signal_Binned_Dataset()
    eval_dataset.load(level, "eval", "../../state/new_physics/data/processed")

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
