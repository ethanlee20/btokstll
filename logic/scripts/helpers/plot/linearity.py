
import numpy
import matplotlib.pyplot as plt

from .util import (
    add_plot_note,
    make_note_model,
    save_plot_model
)


def plot_linearity(
    labels, 
    avgs, 
    stds, 
    config_model,
    path_dir,
    buffer_line_ref=0.05, 
    xlim=(-2.25, 1.35), 
    ylim=(-2.25, 1.35), 
    xlabel=r"Actual $\delta C_9$", 
    ylabel=r"Predicted $\delta C_9$",
):

    _, ax = plt.subplots()

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
    
    add_plot_note(
        ax, 
        note,
    )

    note = make_note_model(
        config_model
    )

    add_plot_note(
        ax, 
        note,
    )

    save_plot_model(
        "lin",
        config_model=config_model,
        path_dir=path_dir,
    )


