
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def setup_high_quality_mpl_params():
    """
    Setup plotting parameters.
    
    i.e. Setup to make fancy looking plots.
    Inspiration from Chris Ketter.

    Side Effects
    ------------
    - Changes matplotlib rc parameters.
    """

    mpl.rcParams["figure.figsize"] = (6, 4)
    mpl.rcParams["figure.dpi"] = 400
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["figure.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["figure.labelsize"] = 30
    mpl.rcParams["xtick.labelsize"] = 12 
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Computer Modern"]
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titley"] = None
    mpl.rcParams["axes.titlepad"] = 4
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["legend.framealpha"] = 0
    mpl.rcParams["legend.markerscale"] = 1
    mpl.rcParams["legend.fontsize"] = 7.5


def make_plot_note(ax, content, fontsize="medium"):
    ax.text(
        1,
        1.01, 
        content, 
        horizontalalignment="right", 
        verticalalignment="bottom", 
        transform=ax.transAxes, 
        fontsize=fontsize
    )


def plot_loss_curves(loss_table, ax, start_epoch=0, log_scale=False):
    """
    Plot loss curves given a loss table.

    Parameters
    ----------
    loss_table : dict
        Dictionary with keys "epoch", "train_loss", and "eval_loss".
    start_epoch : int
        First epoch to plot. (Previous epochs are not plotted.)
    ax : matplotlib axes
        Axes on which to plot.
    """

    epochs_to_plot = loss_table["epoch"][start_epoch:]
    train_losses_to_plot = loss_table["train_loss"][start_epoch:]
    eval_losses_to_plot = loss_table["eval_loss"][start_epoch:]
    
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
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Epoch")


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

    ref_ticks = np.linspace(
        np.min(input_values)-ref_line_buffer, 
        np.max(input_values)+ref_line_buffer, 
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


def plot_volume_slices(arr, n_slices=3, cmap=plt.cm.magma, note=""):
    """
    Plot slices of volumetric data.

    Array dimensions must be (costheta_mu, costheta_K, chi).
    Slices are along the z-axis (axis 2).
    Array arr should be a three-dimensional array.
    Slices might not be evenly spaced along z-axis.
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

    norm = Normalize(vmin=-1.1, vmax=1.1)
    arr = arr.squeeze().cpu()
    colors = cmap(norm(arr))
    
    cartesian_shape = {
        "x": arr.shape[cartesian_dim["x"]],
        "y": arr.shape[cartesian_dim["y"]],
        "z": arr.shape[cartesian_dim["z"]],
    }

    def xy_plane(z_pos):
        x, y = np.indices(
            (cartesian_shape["x"] + 1, cartesian_shape["y"] + 1)
        )
        z = np.full(
            (cartesian_shape["x"] + 1, cartesian_shape["y"] + 1), z_pos
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
        x, y, z = xy_plane(z_index - offset)
        
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            shade=False,
            color="#f2f2f2",
            edgecolor="#f2f2f2", 
        )

    z_indices = np.linspace(0, cartesian_shape["z"]-1, n_slices, dtype=int) # forces integer indices
    for i in z_indices:
        plot_outline(i)
        plot_slice(i)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_3d, location="left", shrink=0.5, pad=-0.05)
    cbar.set_label(r"${q^2}$ (Avg.)", size=11)

    ax_labels = {
        "x": r"$\cos\theta_\mu$",
        "y": r"$\cos\theta_K$",
        "z": r"$\chi$", 
    }
    ax_3d.set_xlabel(ax_labels["x"], labelpad=0)
    ax_3d.set_ylabel(ax_labels["y"], labelpad=0)
    # ax_3d.zaxis.set_rotate_label(False)
    ax_3d.set_zlabel(ax_labels["z"], labelpad=-3,) #rotation="horizontal") 

    ticks = {
        "x": ["-1", "1"],
        "y": ["-1", "1"],
        "z": ['0', r"$2\pi$"],
    }      
    ax_3d.set_xticks([0, cartesian_shape["x"]-1], ticks["x"])
    ax_3d.set_xticks([0, cartesian_shape["y"]-1], ticks["y"])
    ax_3d.set_xticks([0, cartesian_shape["z"]-1], ticks["z"])
    ax_3d.tick_params(pad=0.3)

    ax_3d.set_box_aspect(None, zoom=0.85)

    ax_3d.set_title(f"{note}", loc="center", y=1)