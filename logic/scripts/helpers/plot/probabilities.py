
import matplotlib.pyplot as plt

from ..experiment.constants import delta_C9_value_new_physics, delta_C9_value_standard_model
from .util import add_plot_note, save_model_evaluation_plot


def plot_binned_scatter(ax, x, y, label, color):

    xmin = list(x)
    xmax = list(x[1:])
    xmax.append(xmax[-1]+xmax[1]-xmax[0])
    
    ax.hlines(
        y=y, 
        xmin=xmin,
        xmax=xmax,
        label=label,
        color=color,
        linewidths=2,
    )


def plot_log_probability_distribution_examples(
    log_probabilities, 
    binned_labels, 
    bin_map, 
    model_settings, 
    dataset_settings, 
    path_to_plots_dir
):
    
    unbinned_labels = bin_map[binned_labels]
    example_log_probabilities_standard_model = log_probabilities[unbinned_labels==delta_C9_value_standard_model][0].cpu()
    example_log_probabilities_new_physics = log_probabilities[unbinned_labels==delta_C9_value_new_physics][0].cpu()

    fig, ax = plt.subplots()
    fig.set_figwidth(5)

    ax.axvline(x=delta_C9_value_standard_model, color='black', label=r"$\delta C_9 ="+f"{delta_C9_value_standard_model}$", ls='--')
    ax.axvline(x=delta_C9_value_new_physics, color='black', label=r"$\delta C_9 ="+f"{delta_C9_value_new_physics}$", ls=':')

    plot_binned_scatter(
        x=bin_map.cpu(),
        y=example_log_probabilities_standard_model,
        label=r"SM ($\delta C_9 ="+f"{delta_C9_value_standard_model}$)",
        color='blue',
    )
    plot_binned_scatter(
        x=bin_map.cpu(),
        y=example_log_probabilities_new_physics,
        label=r"NP ($\delta C_9 ="+f"{delta_C9_value_new_physics}$)",
        color='red',
    )

    ax.set_xlabel(r"$\delta C_9$")
    ax.set_ylabel(r"$\log p(\delta C_9 | x_1, ..., x_N)$")
    ax.legend(loc='lower right')
    note = (
        f"Level: {dataset_settings.common.level}"
        f"\nEvents per distribution: {dataset_settings.set.num_events_per_set}"    
    )
    add_plot_note(ax=ax, text=note)
    
    save_model_evaluation_plot(
        type="proba", 
        model_settings=model_settings, 
        dataset_settings=dataset_settings, 
        path_to_plots_dir=path_to_plots_dir
    )
    plt.close()