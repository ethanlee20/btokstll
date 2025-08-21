
from pathlib import Path

from numpy import mean, std
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
    """
    
    ax.text(
        -0.15,
        1.05, 
        text, 
        horizontalalignment="left", 
        verticalalignment="bottom", 
        transform=ax.transAxes, 
        fontsize=fontsize
    )


def save_model_evaluation_plot(type, model_settings, dataset_settings, path_to_plots_dir):
    
    file_name = f"{model_settings.name}_{dataset_settings.common.level}_{dataset_settings.set.num_events_per_set}_{type}.png"
    path = Path(path_to_plots_dir).joinpath(file_name)
    plt.savefig(path, bbox_inches="tight")



def to_x10_notation(number):

    if number == 0:
        return 0
    
    string = f"{number : .1e}"
    e_index = string.find('e')
    string = r"$" + string[:e_index] + r"\times 10^{" + str(int(string[e_index+1:])) + r"}$"
    return string



def stats_legend(
    data_array,
    extra_description=None,
    show_mean=True,
    show_count=True,
    show_stdev=True,
):
    
    """
    Make a legend label similar to the roots stats box.

    Return a string meant to be used as a label for a matplotlib plot.
    Displayable stats are mean, count, and standard deviation.
    """
    
    def calculate_stats(data):
        mean_ = mean(data)
        count = len(data)
        stdev = std(data)
        stats = {
            "mean": mean_,
            "count": count,
            "stdev": stdev,
        }
        return stats
    
    stats = calculate_stats(data_array)

    legend = ""
    if extra_description is not None:
        legend += r"\textbf{" + f"{extra_description}" + "}"
    if show_count:
        legend += f"\nCount: {to_x10_notation(stats['count'])}"
    if show_mean:
        legend += f"\nMean: {to_x10_notation(stats['mean'])}"
    if show_stdev:
        legend += f"\nStdev.: {to_x10_notation(stats['stdev'])}"

    return legend