
from pathlib import Path

import numpy
import matplotlib.pyplot as plt

from .util import add_plot_note, save_model_evaluation_plot


def plot_linearity(
    linearity_test_results,
    model_settings,
    dataset_settings,
    path_to_plots_dir,
    xlim=(-2.25, 1.35), 
    ylim=(-2.25, 1.35),
):
    
    def plot_diagonal_reference_line(unique_labels_numpy_array):
        buffer = 0.05
        ticks = numpy.linspace(
            start=(numpy.min(unique_labels_numpy_array) - buffer),
            stop=(numpy.max(unique_labels_numpy_array) + buffer),
            num=2 
        )
        ax.plot(
            ticks, 
            ticks,
            label="Reference line (slope of 1)",
            color="grey",
            linewidth=0.5,
            zorder=0,
        )  

    unique_labels_numpy_array = linearity_test_results.unique_labels.detach().numpy()
    avgs_numpy_array = linearity_test_results.avgs.detach().numpy()
    stds_numpy_array = linearity_test_results.stds.detach().numpy()

    _, ax = plt.subplots()

    ax.scatter(
        unique_labels_numpy_array, 
        avgs_numpy_array, 
        label="Average over bootstraps", 
        color="firebrick", 
        s=16, 
        zorder=5
    )
    ax.errorbar(
        unique_labels_numpy_array, 
        avgs_numpy_array, 
        yerr=stds_numpy_array, 
        fmt="none", 
        elinewidth=0.5, 
        capsize=0.5, 
        color="black", 
        label="Standard deviation over bootstraps", 
        zorder=10
    )

    plot_diagonal_reference_line(unique_labels_numpy_array)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    ax.set_xlabel(r"Actual $\delta C_9$")
    ax.set_ylabel(r"Predicted $\delta C_9$")
    note = (
        f"Model: {model_settings.name}, Level: {dataset_settings.common.level}", 
        f"\nBootstraps per label: {dataset_settings.set.num_sets_per_label}, Events per bootstrap: {dataset_settings.set.num_events_per_set}"
    )
    add_plot_note(ax=ax, text=note)
    save_model_evaluation_plot(
        type="lin",
        model_settings=model_settings,
        dataset_settings=dataset_settings,
        path_to_plots_dir=path_to_plots_dir
    )
    plt.close()


