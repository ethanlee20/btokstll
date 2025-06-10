

import matplotlib.pyplot as plt

from ..model.loss_table import Loss_Table
from .util import (
    save_plot_model, 
    make_note, 
    add_plot_note
)


def plot_loss_curves(
    config_model,
    config_dset_eval,
    path_dir,
    start_epoch=0, 
    log_scale=False,
    note=None,
):

    loss_table = Loss_Table(
        config_model
        .path_file_loss_table
    )

    epochs_to_plot = (
        loss_table
        .epochs[start_epoch:]
    )

    losses_train_to_plot = (
        loss_table
        .losses_train[start_epoch:]
    )

    losses_eval_to_plot = (
        loss_table
        .losses_eval[start_epoch:]
    )

    _, ax = plt.subplots()

    ax.plot(
        epochs_to_plot, 
        losses_train_to_plot, 
        label="Training Loss"
    )

    ax.plot(
        epochs_to_plot, 
        losses_eval_to_plot, 
        label="Eval. Loss"
    )

    if log_scale:
    
        ax.set_yscale("log")

    ax.legend()

    ax.set_xlabel("Epoch")

    note = make_note(
        config_model=config_model,
        config_dset_eval=config_dset_eval,
    )

    add_plot_note(
        ax, 
        note, 
    )

    save_plot_model(
        "loss_curves",
        config_model,
        config_dset_eval,
        path_dir,
    )
