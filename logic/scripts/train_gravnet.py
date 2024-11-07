
from torch.optim import Adam
from torch.nn import MSELoss

from logic.scripts.library.plotting import setup_mpl_params
from library.modeling.models.gravnet import GravNet_Model
from library.data.datasets.bootstrapped_sets import Bootstrapped_Sets_Dataset
from library.data.datasets.gmm import Gaussian_Mixture_Model_Dataset
from logic.scripts.library.modeling.training import train_and_eval
from library.modeling.lin_test import plot_linearity
from library.modeling.util import select_device, print_gpu_memory_summary


if __name__ == "__main__":

    run_name = "gravnet_gen"
    level = "gen"

    input_dim = 21
    output_dim = 1
    num_blocks = 4
    block_hidden_dim = 64
    block_output_dim = 48
    final_dense_dim = 128
    space_dim = 4
    prop_dim = 22
    k = 40

    model = GravNet_Model(
        input_dim, 
        output_dim, 
        num_blocks,
        block_hidden_dim, 
        block_output_dim,
        final_dense_dim, 
        space_dim, 
        prop_dim, 
        k
    )

    learning_rate = 8e-4
    epochs = 100
    train_batch_size = 1
    eval_batch_size = 1

    device = select_device()

    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataset = Gaussian_Mixture_Model_Dataset()
    train_dataset.load(level, "train", "../../state/new_physics/data/processed", device)
    eval_dataset = Gaussian_Mixture_Model_Dataset()
    eval_dataset.load(level, "eval", "../../state/new_physics/data/processed", device)
    lin_eval_dataset = Gaussian_Mixture_Model_Dataset()
    lin_eval_dataset.load(level, "lin_eval", "../../state/new_physics/data/processed", device)

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

    plot_linearity(run_name, model, lin_eval_dataset, device, plots_dir)
