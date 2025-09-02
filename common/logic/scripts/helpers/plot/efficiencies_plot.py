
import matplotlib.pyplot as plt



def plot_efficiency(
    efficiency_result,
    ax,
    ylimits=(0, 0.5),
):

    ax.scatter(
        efficiency_result.bin_middles, 
        efficiency_result.efficiencies,
        color="purple",
        edgecolors='none',
        s=10,
    )
    ax.errorbar(
        efficiency_result.bin_middles,
        efficiency_result.efficiencies,
        yerr=efficiency_result.errors,
        fmt="none",
        capsize=0,
        elinewidth=0.5, 
        ecolor="black"
    )
    
    ax.set_ylim(ylimits)


# def plot_efficiencies_over_variables(
#     q_squared_efficiency_result,
#     costheta_mu_efficiency_result,
#     costheta_k_efficiency_result,
#     chi_efficiency_result,
#     ylimits=(0, 0.5)
# ):
    
#     def save_plot():
#         pass
    
#     fig, axs = plt.subplots(
#         2, 2, 
#         sharey=True, 
#         layout="compressed"
#     )
    
#     xlimits = {
#         Names_of_Variables().q_squared : (0, 20),
#         Names_of_Variables().costheta_mu : (-1, 1),
#         Names_of_Variables().cos_k : (-1, 1),
#         Names_of_Variables().chi : (0, 2*pi)
#     }

#     xlabels = {
#         Names_of_Variables().q_squared : r"$q^2$ [GeV$^2$]",
#         Names_of_Variables().costheta_mu : r"$\cos\theta_\mu$",
#         Names_of_Variables().cos_k : r"$\cos\theta_K$",
#         Names_of_Variables().chi : r"$\chi$"
#     }   

#     efficiency_results = {
#         Names_of_Variables().q_squared : q_squared_efficiency_result,
#         Names_of_Variables().costheta_mu : costheta_mu_efficiency_result,
#         Names_of_Variables().cos_k : costheta_k_efficiency_result,
#         Names_of_Variables().chi : chi_efficiency_result 
#     }

#     for name_of_variable, ax in zip(Names_of_Variables().tuple_, axs.flat):
#         plot_efficiency(
#             efficiency_result=efficiency_results[name_of_variable],
#             ax=ax,
#             xlabel=xlabels[name_of_variable],
#             xlimits=xlimits[name_of_variable],
#             ylimits=ylimits
#         )

#     save_plot()

