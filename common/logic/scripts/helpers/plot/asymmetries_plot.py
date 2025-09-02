
import matplotlib as mpl
import matplotlib.pyplot as plt

    
def plot_afb(afb_results, ax, cmap, norm, alpha):
    
    for delta_C9_value, bin_middles, afbs, afb_errs in zip(
        afb_results.delta_C9_values,
        afb_results.bin_middles_over_delta_C9_values,
        afb_results.afbs_over_delta_C9_values,
        afb_results.afb_errors_over_delta_C9_values
    ):
        
        is_standard_model = (delta_C9_value == 0)
        color = cmap(norm(delta_C9_value), alpha=alpha) if not is_standard_model else "dimgrey"
        label = None if not is_standard_model else r"SM ($\delta C_9 = 0$)"
        zorder = None if not is_standard_model else 10
        
        ax.scatter(
            bin_middles, 
            afbs, 
            label=label,
            color=color, 
            edgecolors='none',
            s=10,
            zorder=zorder,
        )
        ax.errorbar(
            bin_middles, 
            afbs, 
            yerr=afb_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
            zorder=zorder,
        )

    ax.set_xlabel(r"$q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$A_{FB}$")
    ax.set_ylim(-0.25, 0.46)
    ax.legend()


def plot_s5(s5_results, ax, cmap, norm, alpha):

    for delta_C9_value, bin_middles, s5s, s5_errs in zip(
        s5_results.delta_C9_values,
        s5_results.bin_middles_over_delta_C9_values,
        s5_results.s5s_over_delta_C9_values,
        s5_results.s5_errors_over_delta_C9_values
    ):
        
        is_standard_model = (delta_C9_value == 0)
        color = cmap(norm(delta_C9_value), alpha=alpha) if not is_standard_model else "dimgrey"
        label = None if not is_standard_model else r"SM ($\delta C_9 = 0$)"
        zorder = None if not is_standard_model else 10

        ax.scatter(
            bin_middles, 
            s5s, 
            label=label,
            color=color, 
            edgecolors='none',
            s=10,
            zorder=zorder,
        )
        ax.errorbar(
            bin_middles, 
            s5s, 
            yerr=s5_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
            zorder=zorder,
        )
    
    ax.set_xlabel(r"$q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$S_{5}$")
    ax.set_ylim(-0.48, 0.38)
    ax.legend()



def plot_afb_and_s5(fig, axs_2x1, afb_results, s5_results, alpha=0.85):

    def get_colorbar_halfrange(afb_results, s5_results):
        assert min(afb_results.delta_C9_values) == min(s5_results.delta_C9_values)
        return abs(min(afb_results.delta_C9_values))
   
    afb_ax = axs_2x1.flat[0]
    s5_ax = axs_2x1.flat[1]

    cmap = plt.cm.coolwarm
    norm = mpl.colors.CenteredNorm(
        vcenter=0, 
        halfrange=get_colorbar_halfrange(afb_results=afb_results, s5_results=s5_results),
    )

    plot_afb(afb_results=afb_results, ax=afb_ax, cmap=cmap, norm=norm, alpha=alpha)
    plot_s5(s5_results=s5_results, ax=s5_ax, cmap=cmap, norm=norm, alpha=alpha)

    s5_ax.get_legend().remove()
    afb_ax.get_xlabel().remove()

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=axs_2x1, 
        orientation='vertical', 
        label=r'$\delta C_9$',
    )