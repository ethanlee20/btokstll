
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


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


def plot_loss_curves(epochs:list, train_losses:list, eval_losses:list, ax):
    ax.plot(epochs, train_losses, label="Training Loss")
    ax.plot(epochs, eval_losses, label="Eval. Loss")
    ax.legend()
    ax.set_xlabel("Epoch")


def plot_likelihood_over_bins(predictions_over_bins, bin_values, cmap=plt.cm.viridis):
    """
    predictions_over_bins : ndarray of summed log event probabilities
        (rows are input bins, columns are bin predictions)
    bin_values : ndarray of the value each bin represents 
    """

    fig, ax = plt.subplots(layout="constrained")

    bounds = np.append(bin_values, bin_values[-1] + (bin_values[-1] - bin_values[-2]))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    for b_v, pred in zip(bin_values, predictions_over_bins):
        pred_bin = np.argmax(pred)
        ax.plot(bin_values, pred, color=cmap(norm(b_v)))
        ax.scatter(bin_values[pred_bin], np.max(pred), color=cmap(norm(b_v)), edgecolors="black", zorder=100)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$\delta C_9$")
    ax.set_xlabel(r"$\delta C_9$")
    ax.set_ylabel(r"$\sum_i \log p(\delta C_9 \;|\; x_i)$")

    plt.show()   


def plot_prediction_linearity(
        input_values, avg_pred, stdev_pred, ref_line_buffer, 
        xlim=None, ylim=None, xlabel=None, ylabel=None
):
    """
    input_values : value corresponding to each bin index
    avg_pred : ndarray of average prediction per input bin
    stdev_pred : ndarray of standard deviation of prediction per input bin 
    ref_line_buffer : extra amount to extend reference line
    xlim : x limits
    ylim : y limits
    """
    _, ax = plt.subplots()
        
    ax.scatter(input_values, avg_pred, label="Validation Results", color="firebrick", s=16, zorder=5)
    ax.errorbar(input_values, avg_pred, yerr=stdev_pred, fmt="none", elinewidth=0.5, capsize=0.5, color="black", label="Std. Dev.", zorder=10)

    ref_ticks = np.linspace(np.min(input_values)-ref_line_buffer, np.max(input_values)+ref_line_buffer, 2)
    ax.plot(
        ref_ticks, ref_ticks,
        label="Ref. Line (Slope = 1)",
        color="grey",
        linewidth=0.5,
        zorder=0
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()
    if xlabel is not None:
        ax.set_xlabel(xlabel) # )
    if ylabel is not None:
        ax.set_ylabel(ylabel) # r

    plt.show()