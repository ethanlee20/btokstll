
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_high_quality_mpl_params():

    """
    Setup plotting parameters.
    
    Setup environment to make 
    fancy looking plots.
    Inspiration from Chris Ketter.
    Good quality for exporting.

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


def add_plot_note(ax, text:str, fontsize="medium"):

    """
    Annotate a plot in the upper right corner,
    above the plot box.

    This doesn't work for 3D plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    text : str
        What to write.
    fontsize : str
        The fontsize.
        Takes any argument that the
        axes.text() method can handle.
    
    Side Effects
    ------------
    - Modifies the given axes.
    """

    ax.text(
        1,
        1.01, 
        text, 
        horizontalalignment="right", 
        verticalalignment="bottom", 
        transform=ax.transAxes, 
        fontsize=fontsize
    )


def save_plot_model(
    kind, 
    config_model,
    path_dir,
):

    config_dset_train = config_model.config_dset_train

    name_file = (
        f"{config_model.name}_"
        f"{config_dset_train.num_events_per_set}_"
        f"{config_dset_train.level}_"
        f"q2v_{config_dset_train.q_squared_veto}_"
        f"{kind}"
        ".png"
    )

    path_dir = pathlib.Path(
        path_dir
    )

    path_file = path_dir.joinpath(
        name_file
    )

    plt.savefig(
        path_file, 
        bbox_inches="tight"
    )

    plt.close()


def save_plot_dset(
    kind, 
    config_dset,
    path_dir
):

    name_file = (
        f"{config_dset.name}_"
        f"{config_dset.num_events_per_set}_"
        f"{config_dset.level}_"
        f"{config_dset.split}_"
        f"{kind}"
        ".png"
    )

    path_dir = pathlib.Path(
        path_dir
    )

    path_file = path_dir.joinpath(
        name_file
    )

    plt.savefig(
        path_file, 
        bbox_inches="tight"
    )

    plt.close()


def make_note_model(
    config_model
):
    
    config_dset_train = config_model.config_dset_train

    note = (
        f"{config_model.name}, "
        f"{config_dset_train.level}., "
        f"{config_dset_train.num_sets_per_label} boots., "
        f"{config_dset_train.num_events_per_set} events/boots."
    )

    return note