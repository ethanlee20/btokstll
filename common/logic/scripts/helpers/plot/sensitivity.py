
import matplotlib.pyplot as plt

from .util import add_plot_note, save_model_evaluation_plot


def plot_sensitivity(
    sensitivity_test_results,
    model_settings,
    dataset_settings,
    path_to_plots_dir,
    bins=50, 
    xbounds=(-1.5, 0), 
    ybounds=(0, 200), 
):
    
    def add_label_marker(ax, label, ybounds):
        ax.vlines(
            x=label,
            ymin=ybounds[0],
            ymax=ybounds[1],
            color="red",
            label=r"Actual $\delta C_9$ " + f"({label})"
        )

    def add_prediction_average_marker(ax, average, ybounds):
        ax.vlines(
            x=average,
            ymin=ybounds[0],
            ymax=ybounds[1],
            color="red",
            linestyles="--",
            label=f"Average: {average.round(decimals=3)}"
        )

    def add_prediction_standard_deviation_marker(ax, average, standard_deviation):
        ax.hlines(
            y=(ybounds[1] / 10),
            xmin=average,
            xmax=average+standard_deviation,
            color="orange",
            linestyles="dashdot",
            label=(f"Standard deviation: {standard_deviation.round(decimals=3)}")
        )

    _, ax = plt.subplots()

    ax.hist(sensitivity_test_results.predicted_values, bins=bins, range=xbounds)

    add_label_marker(ax=ax, label=sensitivity_test_results.label, ybounds=ybounds)
    add_prediction_average_marker(ax=ax, average=sensitivity_test_results.avg, ybounds=ybounds)
    add_prediction_standard_deviation_marker(
        ax=ax, 
        average=sensitivity_test_results.avg, 
        standard_deviation=sensitivity_test_results.std
    )
    ax.set_xlabel(r"Predicted $\delta C_9$")
    ax.set_xbound(*xbounds)
    ax.set_ybound(*ybounds)
    ax.legend()
    note = (
        f"Model: {model_settings.name}, Level: {dataset_settings.common.level}", 
        f"\nNumber of bootstraps: {dataset_settings.set.num_sets_per_label}, Events per bootstrap: {dataset_settings.set.num_events_per_set}"
    )
    add_plot_note(ax=ax, text=note)
    
    save_model_evaluation_plot(
        type="sens",
        model_settings=model_settings,
        dataset_settings=dataset_settings,
        path_to_plots_dir=path_to_plots_dir
    )
    plt.close()

    


