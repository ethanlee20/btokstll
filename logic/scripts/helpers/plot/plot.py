
"""
Plotting
"""

import pathlib

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..model.config import Config_Model
from ..model.loss_table import Loss_Table
from ..data.dset.constants import Names_Datasets
from .util import make_plot_note


class Plotter:
    
    def __init__(
        self,
        path_dir,
        config_model=None,
        config_dset=None,
    ):

        """
        Specify config_model xor config_dset.
        """
 
        if (
            (config_dset is not None) 
            and (config_model is not None)
        ):
            
            raise ValueError(
                "Specify config_dset xor config_model."
            )

        self.path_dir = pathlib.Path(
            path_dir
        )

        self.config_model = config_model

        if config_model:

            self.config_dset = config_model.config_dset

        else:

            self.config_dset = config_dset

        if self.config_dset is None:

            raise ValueError(
                "Specify config_dset xor config_model."
            )

    def plot_image_slices(
        self, 
        image, 
    ):
        
        fig = plt.figure()

        ax_3d = fig.add_subplot(
            projection="3d"
        )

        note = (
            "Events per set: "
            f"{self.config_dset.num_events_per_set}\n"
            "Bins per dim.: "
            f"{self.config_dset.num_bins_image}"
        )

        _plot_image_slices(
            fig, 
            ax_3d, 
            image, 
            note=note
        )

        self._save_plot_and_close(
            "slices_image"
        )

    def plot_loss_curves(
        self,
        start_epoch=0, 
        log_scale=True,
    ):

        self._check_config_model_specified()

        _, ax = plt.subplots()

        note = self._make_note_model()

        loss_table = Loss_Table(
            self.config_model
            .path_file_loss_table
        )

        _plot_loss_curves(
            ax, 
            loss_table,
            start_epoch=start_epoch, 
            log_scale=log_scale,
            note=note
        )

        self._save_plot_and_close(
            "loss_curves"
        )

    def plot_sensitivity(
        self,
        preds,
        avg,
        std,
        label,
        bins=50, 
        xbounds=(-1.5, 0), 
        ybounds=(0, 200), 
    ):
        
        self._check_config_model_specified()
        
        _, ax = plt.subplots()

        note = self._make_note_model()

        _plot_sensitivity(
            ax,
            preds,
            avg,
            std,
            label,
            bins=bins,
            xbounds=xbounds,
            ybounds=ybounds,
            note=note,
        )

        self._save_plot_and_close(
            "sens"
        )

    def plot_linearity(
        self,
        labels,
        avgs,
        stds,
    ):

        self._check_config_model_specified()

        _, ax = plt.subplots()

        note = self._make_note_model()

        _plot_linearity(
            ax,
            labels, 
            avgs, 
            stds, 
            note=note,
        )

        self._save_plot_and_close(
            "lin"
        )

    def _save_plot_and_close(self, kind):

        name_file = self._make_name_file_plot(
            kind
        )

        path_file = self.path_dir.joinpath(
            name_file
        )
        
        plt.savefig(
            path_file, 
            bbox_inches="tight"
        )

        plt.close()
    
    def _make_name_file_plot(self, kind):
        
        if self.config_model is None:

            name = self._make_name_file_plot_dset(kind)

        else:
            
            name = self._make_name_file_plot_model(kind)

        return name

    def _make_name_file_plot_model(self, kind):

        name = (
            f"{self.config_model.name}_"
            f"{self.config_dset.num_events_per_set}_"
            f"{self.config_dset.level}_"
            f"q2v_{self.config_dset.q_squared_veto}_"
            f"{kind}"
            ".png"
        )

        return name
    
    def _make_name_file_plot_dset(self, kind):

        name = (
            f"{self.config_dset.name}_"
            f"{self.config_dset.num_events_per_set}_"
            f"{self.config_dset.split}_"
            f"{kind}"
            ".png"
        )

        return name
    
    def _make_note_model(self):
        
        note = (
            f"{self.config_model.name}, "
            f"{self.config_dset.level}., "
            f"{self.config_dset.num_sets_per_label} boots., "
            f"{self.config_dset.num_events_per_set} events/boots."
        )

        return note
    
    def _check_config_model_specified(self):
        
        if self.config_model is None:

            raise ValueError(
                "config_model must be specified."
            )


def _plot_image_slices(
    fig,
    ax_3d,
    image, 
    n_slices=3, 
    cmap=plt.cm.magma, 
    note=None,
):
    
    """
    Plot slices of a B->K*ll dataset image.

    Slices are along the chi-axis (axis 2) 
    and might not be evenly spaced.
    """

    cartesian_dim = {
        "x": 0,     
        "y": 1,
        "z": 2,  
    }

    norm = mpl.colors.Normalize(
        vmin=-1.1, 
        vmax=1.1,
    )

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

    # forces integer indices
    z_indices = numpy.linspace( 
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

    ax_3d.set_title(
        f"{note}", 
        loc="center", 
        y=1
    )


def _plot_loss_curves(
    ax,
    loss_table,
    start_epoch=0, 
    log_scale=False,
    note=None,
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

    make_plot_note(
        ax, 
        note, 
    )


def _plot_sensitivity(
    ax,
    preds,
    avg,
    std,
    label,
    bins=50, 
    xbounds=(-1.5, 0), 
    ybounds=(0, 200), 
    note=None,
):
    
    ax.hist(
        preds, 
        bins=bins, 
        range=xbounds,
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
        label=(
            r"$\mu = $ " 
            + f"{avg.round(decimals=3)}"
        ),
    )

    height_marker_std = (
        ybounds[1] 
        / 10
    )

    ax.hlines(
        height_marker_std,
        avg,
        avg+std,
        color="orange",
        linestyles="dashdot",
        label=(
            r"$\sigma = $ " 
            + f"{std.round(decimals=3)}"
        ),
    )

    ax.set_xlabel(
        r"Predicted $\delta C_9$"
    )

    ax.set_xbound(*xbounds)
    
    ax.set_ybound(*ybounds)
    
    ax.legend()
    
    make_plot_note(
        ax, 
        note, 
        fontsize="medium"
    )


def _plot_linearity(
    ax,
    labels, 
    avgs, 
    stds, 
    buffer_line_ref=0.05, 
    xlim=(-2.25, 1.35), 
    ylim=(-2.25, 1.35), 
    xlabel=r"Actual $\delta C_9$", 
    ylabel=r"Predicted $\delta C_9$",
    note=None,
):
        
    ax.scatter(
        labels, 
        avgs, 
        label="Avg.", 
        color="firebrick", 
        s=16, 
        zorder=5
    )

    ax.errorbar(
        labels, 
        avgs, 
        yerr=stds, 
        fmt="none", 
        elinewidth=0.5, 
        capsize=0.5, 
        color="black", 
        label="Std. Dev.", 
        zorder=10
    )

    ref_ticks = numpy.linspace(
        (
            numpy.min(labels)
            - buffer_line_ref
        ), 
        (
            numpy.max(labels)
            + buffer_line_ref
        ), 
        2,
    )

    ax.plot(
        ref_ticks, 
        ref_ticks,
        label="Ref. Line",
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
    
    make_plot_note(
        ax, 
        note,
    )

