

import matplotlib.pyplot as plt

from ..models.loss_table import Loss_Table
from ..models.constants import Names_of_Models
from .util import add_plot_note


def plot_loss_curves(
    model_settings,
    start_epoch=0, 
    log_scale=False,
    note=None,
):
    
    loss_table = Loss_Table()
    loss_table.load(model_settings.paths.make_path_to_loss_table_file())

    epochs_to_plot = loss_table.epochs[start_epoch:]
    train_losses_to_plot = loss_table.train_losses[start_epoch:]
    eval_losses_to_plot = loss_table.eval_losses[start_epoch:]

    _, ax = plt.subplots()

    ax.plot(
        epochs_to_plot, 
        train_losses_to_plot, 
        label="Training Loss"
    )
    ax.plot(
        epochs_to_plot, 
        eval_losses_to_plot, 
        label="Eval. Loss"
    )

    if log_scale: ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Epoch")
    note = (
        (
            f"Loss curves: {model_settings.name}, {model_settings.training.training_dataset_settings.common.level}, ", 
            f"events/set: {model_settings.training.training_dataset_settings.set.num_events_per_set}"
        )
        if model_settings.name in Names_of_Models().set_based
        else f"Loss curves: {model_settings.name}, {model_settings.training.training_dataset_settings.common.level}"
        if model_settings.name in Names_of_Models().event_based
        else None
    )
    if note is None: raise ValueError
    add_plot_note(ax=ax, text=note)
    
    plt.savefig(model_settings.paths.make_path_to_loss_curves_plot(), bbox_inches="tight")
    plt.close()
