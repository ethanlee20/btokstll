
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss

from helpers.modeling.models.cnn import CNN
from helpers.data.datasets.histogram_image import Histogram_Image_Dataset
from helpers.modeling.train import train_and_eval
from helpers.modeling.lin_test import plot_linearity
from helpers.modeling.util import select_device


run_name = "cnn_test_det"
level = "det"
model = CNN()

learning_rate = 1e-3
epochs = 20 
train_batch_size = 20
eval_batch_size = 15

device = select_device()

loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

train_dataset = Histogram_Image_Dataset()
train_dataset.load(level, "train", "../../state/new_physics/data/processed", device)
eval_dataset = Histogram_Image_Dataset()
eval_dataset.load(level, "eval", "../../state/new_physics/data/processed", device)

models_dir = "../../state/new_physics/models"
plots_dir = "../../state/new_physics/plots"


train_and_eval(
    run_name, model, train_dataset, eval_dataset,
    loss_fn, optimizer, device,
    models_dir, plots_dir,
    epochs, train_batch_size, eval_batch_size
)

plot_linearity(run_name, model, eval_dataset, device, plots_dir)