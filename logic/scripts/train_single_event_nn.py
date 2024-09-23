
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss

from library.modeling.models.single_event_nn import Single_Event_NN
from library.data.datasets.aggregated_raw import Aggregated_Raw_Dataset
from library.modeling.train import train_and_eval
from library.modeling.lin_test import plot_linearity
from library.modeling.util import select_device


run_name = "single_event_nn"
level = "gen"
model = Single_Event_NN()

learning_rate = 1e-3
epochs = 200
train_batch_size = 24_000
eval_batch_size = 24_000

device = select_device()

loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

label = "dc9"
train_dataset = Aggregated_Raw_Dataset()
train_dataset.load(level, "train", label, "../../state/new_physics/data/processed")
eval_dataset = Aggregated_Raw_Dataset()
eval_dataset.load(level, "eval", label, "../../state/new_physics/data/processed")

models_dir = "../../state/new_physics/models"
plots_dir = "../../state/new_physics/plots"

train_and_eval(
    run_name, model, train_dataset, eval_dataset,
    loss_fn, optimizer, device,
    models_dir, plots_dir,
    epochs, train_batch_size, eval_batch_size
)

plot_linearity(run_name, model, eval_dataset, device, plots_dir)