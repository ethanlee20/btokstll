
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss

from library.modeling.models.deep_sets import Deep_Sets
from library.data.datasets.bootstrapped_sets import Bootstrapped_Sets_Dataset
from library.modeling.train import train_and_eval
from library.modeling.lin_test import plot_linearity
from library.modeling.util import select_device


run_name = "deep_sets_gen"
level = "gen"
model = Deep_Sets()

learning_rate = 1e-3
epochs = 500
train_batch_size = 64
eval_batch_size = 64

device = select_device()

loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

train_dataset = Bootstrapped_Sets_Dataset()
train_dataset.load(level, "train", "../../state/new_physics/data/processed")
eval_dataset = Bootstrapped_Sets_Dataset()
eval_dataset.load(level, "eval", "../../state/new_physics/data/processed")

models_dir = "../../state/new_physics/models"
plots_dir = "../../state/new_physics/plots"

train_and_eval(
    run_name, model, train_dataset, eval_dataset,
    loss_fn, optimizer, device,
    models_dir, plots_dir,
    epochs, train_batch_size, eval_batch_size
)

plot_linearity(run_name, model, eval_dataset, device, plots_dir)