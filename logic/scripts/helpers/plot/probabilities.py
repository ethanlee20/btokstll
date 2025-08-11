
import matplotlib.pyplot as plt

from ..experiment.constants import delta_C9_value_new_physics, delta_C9_value_standard_model
from .util import add_plot_note, save_model_evaluation_plot


def plot_log_probability_distribution_examples(
    log_probabilities, 
    binned_labels, 
    bin_map,
    ax
):

    unbinned_labels = bin_map[binned_labels]
    example_log_probabilities_standard_model = log_probabilities[unbinned_labels==delta_C9_value_standard_model][0].cpu()
    example_log_probabilities_new_physics = log_probabilities[unbinned_labels==delta_C9_value_new_physics][0].cpu()

    ax.axvline(x=delta_C9_value_standard_model, color='black', label=None, ls='--')
    ax.axvline(x=delta_C9_value_new_physics, color='black', label=None, ls=':')

    ax.scatter(bin_map.cpu(), example_log_probabilities_standard_model, label="SM", color="blue", s=2.5, marker="s")
    ax.scatter(bin_map.cpu(), example_log_probabilities_new_physics, label="NP", color="red", s=3, marker="o")

    ax.set_xticks([-2, -0.82, 0, 1.1])