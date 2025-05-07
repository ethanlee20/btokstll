
import matplotlib as mpl


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


def make_plot_note(ax, text:str, fontsize="medium"):

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