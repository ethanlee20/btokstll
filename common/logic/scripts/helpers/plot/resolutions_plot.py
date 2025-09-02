
import matplotlib.pyplot as plt

import stats_legend
import Names_of_Variables


def plot_resolution(
    resolution,
    name_of_variable,
    xlimits,
    num_bins
):
    
    def save_plot():
        pass
        
    fig, ax = plt.subplots()

    ax.hist(
        resolution,
        label=stats_legend(resolution),
        bins=num_bins,
        range=xlimits,
        color="purple",
        histtype="step",
    )

    ax.legend()
    ax.set_xlim(xlimits)

    xlabels = {
        Names_of_Variables().q_squared : r"$q^2 - q^{2\;MC}$",
        Names_of_Variables().costheta_mu : r"$\cos\theta_\mu - \cos\theta_\mu^{MC}$",
        Names_of_Variables().cos_k : r"$\cos\theta_K - \cos\theta_K^{MC}$",
        Names_of_Variables().chi : r"$\chi - \chi^{MC}$"
    }
    ax.set_xlabel(xlabels[name_of_variable])

    save_plot()