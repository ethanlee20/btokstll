
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..datasets.constants import Names_of_Labels, Names_of_Variables


def plot_distribution(data, ax, delta_C9_value, cmap, norm, xlimits, num_bins):

    is_standard_model = (delta_C9_value == 0)

    color = cmap(norm(delta_C9_value)) if not is_standard_model else "dimgrey"
    linestyle = 'solid' if not is_standard_model else (0, (1, 1))
    label = None if not is_standard_model else r"SM ($\delta C_9 = 0$)"
    zorder = None if not is_standard_model else 10
    
    ax.hist(
        data, 
        range=xlimits,
        histtype="step",
        density=True, 
        bins=num_bins, 
        color=color,
        linestyle=linestyle,
        label=label,
        zorder=zorder
    )


def plot_variable(dataframe, name_of_variable, ax, cmap, norm, num_bins):

    assert name_of_variable in Names_of_Variables().tuple_
    
    xlabels = {
        Names_of_Variables().q_squared : r"$q^2$ [GeV$^2$]",
        Names_of_Variables().cos_theta_mu : r"$\cos\theta_\mu$",
        Names_of_Variables().cos_k : r"$\cos\theta_K$",
        Names_of_Variables().chi : r"$\chi$"
    }
    xlimits = {
        Names_of_Variables().q_squared : (0, 20),
        Names_of_Variables().cos_theta_mu : (-1, 1),
        Names_of_Variables().cos_k : (-1, 1),
        Names_of_Variables().chi : (0, 2*pi),
    }
    ylims = {
        Names_of_Variables().q_squared : (0, 0.14),
        Names_of_Variables().cos_theta_mu : (0, 0.75),
        Names_of_Variables().cos_k : (0, 0.85),
        Names_of_Variables().chi : (0, 0.195),
    }

    for delta_C9_value, dataframe_subset in dataframe.groupby(Names_of_Labels().unbinned):
        plot_distribution(
            data=dataframe_subset[name_of_variable],
            ax=ax,
            delta_C9_value=delta_C9_value,
            cmap=cmap,
            norm=norm,
            xlimits=xlimits[name_of_variable],
            num_bins=num_bins
        )

    if name_of_variable == Names_of_Variables().chi:
        ax.set_xticks(
            [0, pi, 2*pi], 
            ["0", r"$\pi$", r"$2 \pi$"],
        )
    ax.set_xlabel(xlabels[name_of_variable])
    ax.set_ylim(ylims[name_of_variable])
    ax.locator_params(axis='y', tight=True, nbins=2)


def plot_all_variables(fig, axs_2x2, dataframe, num_bins):

    cmap = plt.cm.coolwarm
    norm = mpl.colors.CenteredNorm(
        vcenter=0, 
        halfrange=abs(dataframe[Names_of_Labels().unbinned].min())
    )

    for variable, ax in zip(Names_of_Variables().tuple_, axs_2x2.flat):
        plot_variable(
            dataframe=dataframe,
            name_of_variable=variable,
            ax=ax,
            cmap=cmap,
            norm=norm,
            num_bins=num_bins
        )

    axs_2x2[0,0].legend()
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=axs_2x2, 
        orientation='vertical', 
    )
    cbar.set_label(r'$\delta C_9$', size=11)