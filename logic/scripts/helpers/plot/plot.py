
"""
Plotting
"""

import pathlib

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..model.config import Config_Model
from ..model.loss_table import Loss_Table
from .util import make_plot_note



def plot_image_slices(
    image, 
    config_dset,
    n_slices=3, 
    cmap=plt.cm.magma, 
):
    
    """
    Plot slices of a B->K*ll dataset image.

    Slices are along the chi-axis (axis 2) and might
    not be evenly spaced.
    """

    fig = plt.figure()

    ax_3d = fig.add_subplot(projection="3d")

    var_dim = {
        0: "costheta_mu",
        1: "costheta_K",
        2: "chi",
    }

    cartesian_dim = {
        "x": 0,     
        "y": 1,
        "z": 2,  
    }

    norm = mpl.colors.Normalize(vmin=-1.1, vmax=1.1)

    image = image.squeeze().cpu()

    colors = cmap(norm(image))
    
    cartesian_shape = {
        "x": image.shape[cartesian_dim["x"]],
        "y": image.shape[cartesian_dim["y"]],
        "z": image.shape[cartesian_dim["z"]],
    }

    def xy_plane(z_pos):

        x, y = numpy.indices(
            (
                cartesian_shape["x"] + 1, 
                cartesian_shape["y"] + 1
            )
        )

        z = numpy.full(
            (
                cartesian_shape["x"] + 1, 
                cartesian_shape["y"] + 1,
            ), z_pos
        )

        return x, y, z
    
    def plot_slice(z_index):

        x, y, z = xy_plane(z_index) 

        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            facecolors=colors[:,:,z_index], 
            shade=False,
        )

    def plot_outline(z_index, offset=0.3):

        x, y, z = xy_plane(
            z_index - offset
        )
        
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, 
            cstride=1, 
            shade=False,
            color="#f2f2f2",
            edgecolor="#f2f2f2", 
        )

    z_indices = numpy.linspace( # forces integer indices
        0, 
        cartesian_shape["z"]-1, 
        n_slices, 
        dtype=int
    ) 
    
    for i in z_indices:

        plot_outline(i)

        plot_slice(i)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=norm, 
            cmap=cmap
        ), 
        ax=ax_3d, 
        location="left", 
        shrink=0.5, 
        pad=-0.05
    )

    cbar.set_label(
        r"${q^2}$ (Avg.)", 
        size=11
    )

    ax_labels = {
        "x": r"$\cos\theta_\mu$",
        "y": r"$\cos\theta_K$",
        "z": r"$\chi$", 
    }

    ax_3d.set_xlabel(
        ax_labels["x"], 
        labelpad=0
    )

    ax_3d.set_ylabel(
        ax_labels["y"], 
        labelpad=0
    )

    ax_3d.set_zlabel(
        ax_labels["z"], 
        labelpad=-3
    )

    ticks = {
        "x": ["-1", "1"],
        "y": ["-1", "1"],
        "z": ['0', r"$2\pi$"],
    }      

    ax_3d.set_xticks(
        [
            0, 
            cartesian_shape["x"]-1
        ], 
        ticks["x"]
    )
    
    ax_3d.set_xticks(
        [
            0, 
            cartesian_shape["y"]-1
        ], 
        ticks["y"]
    )
    
    ax_3d.set_xticks(
        [
            0, 
            cartesian_shape["z"]-1
        ], 
        ticks["z"]
    )
    
    ax_3d.tick_params(pad=0.3)
    
    ax_3d.set_box_aspect(
        None, 
        zoom=0.85
    )

    note = (
        "Events per set: "
        f"{config_dset.num_events_per_set}\n"
        "Bins per dim.: "
        f"{config_dset.num_bins_image}"
    )

    ax_3d.set_title(
        f"{note}", 
        loc="center", 
        y=1
    )

    name_file = (
        f"{config_dset.num_events_per_set}_"
        f"{config_dset.split}_"
        "image_slices.png"
    )

    save_plot(
        config_dset.path_dir,
        name_file
    )


def plot_loss_curves(
    config_model,
    start_epoch=0, 
    log_scale=False,
):

    """
    Plot loss curves.

    Parameters
    ----------
    loss_table : Loss_Table
    start_epoch : int
        Start plotting from here. 
    ax : matplotlib.Axes
        PLot on this axes.
    """

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

    name_file = (
        "loss_curves.png"
    )

    save_plot(
        config_model.path_dir,
        name_file,
    )


def plot_sensitivity(
    preds,
    avg,
    std,
    label,
    config_model:Config_Model,
    bins=50, 
    xbounds=(-1.5, 0), 
    ybounds=(0, 200), 
    std_marker_height=20,
    note=None,
):
    
    _, ax = plt.subplots()
    
    ax.hist(
        preds, 
        bins=bins, 
        range=xbounds
    )

    ax.vlines(
        label,
        0,
        ybounds[1],
        color="red",
        label=f"Target ({label})",
    )

    ax.vlines(
        avg,
        0,
        ybounds[1],
        color="red",
        linestyles="--",
        label=r"$\mu = $ " + f"{avg.round(decimals=3)}"
    )

    ax.hlines(
        std_marker_height,
        avg,
        avg+std,
        color="orange",
        linestyles="dashdot",
        label=r"$\sigma = $ " + f"{std.round(decimals=3)}"
    )

    ax.set_xlabel(r"Predicted $\delta C_9$")

    ax.set_xbound(*xbounds)
    
    ax.set_ybound(*ybounds)
    
    ax.legend()
    
    make_plot_note(
        ax, 
        note, 
        fontsize="medium"
    )

    name_file = (
        f"sens_"
        f"{
            config_model.config_dset
            .num_sets_per_label
        }"
        ".png"
    )

    save_plot(
        config_model.path_dir,
        name_file,
    )


def save_plot(path_dir, name_file):

    path = path_dir.join(
        name_file
    )

    plt.savefig(path, bbox_inches="tight")

    print(
        f"Saved plot to file: {path}"
    )

    plt.show()
    
    plt.close()













def plot_prediction_linearity(
    ax,
    input_values, 
    avg_pred, 
    stdev_pred, 
    ref_line_buffer=0.05, 
    xlim=(-2.25, 1.35), 
    ylim=(-2.25, 1.35), 
    xlabel=r"Actual $\delta C_9$", 
    ylabel=r"Predicted $\delta C_9$",
    note=None,
):
    """
    input_values : array 
        value corresponding to each bin index
    avg_pred : array
        ndarray of average prediction per input bin
    stdev_pred : array
        ndarray of standard deviation of prediction per input bin 
    ref_line_buffer : float 
        extra amount to extend reference line
    xlim : tuple
        x limits
    ylim : tuple
        y limits
    xlabel : str
        x axis label
    ylabel : str
        y axis label
    note : str
        note to add
    """
        
    ax.scatter(
        input_values, 
        avg_pred, 
        label="Avg.", 
        color="firebrick", 
        s=16, 
        zorder=5
    )
    ax.errorbar(
        input_values, 
        avg_pred, 
        yerr=stdev_pred, 
        fmt="none", 
        elinewidth=0.5, 
        capsize=0.5, 
        color="black", 
        label="Std. Dev.", 
        zorder=10
    )

    ref_ticks = numpy.linspace(
        numpy.min(input_values)-ref_line_buffer, 
        numpy.max(input_values)+ref_line_buffer, 
        2,
    )
    ax.plot(
        ref_ticks, 
        ref_ticks,
        label="Ref. Line (Slope = 1)",
        color="grey",
        linewidth=0.5,
        zorder=0,
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    make_plot_note(ax, note)

