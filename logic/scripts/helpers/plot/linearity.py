
import numpy


def plot_linearity(
    linearity_test_results,
    xlim=(-2.25, 1.35), 
    ylim=(-2.25, 1.35),
    ax=None,
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
            label=None,
            color="grey",
            linewidth=0.5,
            zorder=0,
        )  

    unique_labels_numpy_array = linearity_test_results.unique_labels.cpu().detach().numpy()
    avgs_numpy_array = linearity_test_results.avgs.cpu().detach().numpy()
    stds_numpy_array = linearity_test_results.stds.cpu().detach().numpy()

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        unique_labels_numpy_array, 
        avgs_numpy_array, 
        label="Avg.", 
        color="firebrick", 
        s=6, 
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
        label="Stdev.", 
        zorder=10
    )

    plot_diagonal_reference_line(unique_labels_numpy_array)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks(ticks=[-2, 0, 1.1])
    ax.set_yticks(ticks=[-2, 0, 1.1])