

from pathlib import Path

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..experiment.constants import delta_C9_value_new_physics, delta_C9_value_standard_model


def plot_image_examples(dataset, path_to_plots_dir):

    example_image_standard_model = dataset.features[dataset.labels==delta_C9_value_standard_model][0]
    example_image_new_physics = dataset.features[dataset.labels==delta_C9_value_new_physics][0]
    
    plot_image_slices(
        image=example_image_standard_model,
        dataset_settings=dataset.settings,
        delta_C9_value=delta_C9_value_standard_model,
        path_to_plots_dir=path_to_plots_dir
    )
    plot_image_slices(
        image=example_image_new_physics,
        dataset_settings=dataset.settings,
        delta_C9_value=delta_C9_value_new_physics,
        path_to_plots_dir=path_to_plots_dir
    )


def plot_image_slices(
    image, 
    dataset_settings,
    delta_C9_value,
    path_to_plots_dir,
    num_slices=3, 
    cmap=plt.cm.magma
):  
    
    """
    Plot slices of a B->K*ll dataset image.

    Slices are along the chi-axis (axis 2)
    and might not be evenly spaced.
    """

    def xy_plane_at(z_position):
        x, y = numpy.indices(
            (
                axis_dimension_from_cartesian["x"] + 1, 
                axis_dimension_from_cartesian["y"] + 1
            )
        )
        z = numpy.full(
            (
                axis_dimension_from_cartesian["x"] + 1, 
                axis_dimension_from_cartesian["y"] + 1
            ), 
            z_position
        )
        return x, y, z
    
    def plot_slice(z_index):
        x, y, z = xy_plane_at(z_index) 
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            facecolors=colors[:,:,z_index], 
            shade=False
        )

    def plot_outline(z_index, offset=0.3):
        x, y, z = xy_plane_at(z_index - offset)
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, 
            cstride=1, 
            shade=False,
            color="#f2f2f2",
            edgecolor="#f2f2f2"
        )

    def save_plot(dataset_settings, path_to_plots_dir):
        file_name = f"image_{dataset_settings.common.level}_{dataset_settings.set.num_events_per_set}.png"
        path = Path(path_to_plots_dir).joinpath(file_name)
        plt.savefig(path, bbox_inches="tight")

    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d")

    image = image.squeeze().cpu()
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    colors = cmap(norm(image))
    
    axis_index_from_cartesian = {
        "x": 0,
        "y": 1,
        "z": 2
    }
    axis_dimension_from_cartesian = {
        "x": image.shape[axis_index_from_cartesian["x"]],
        "y": image.shape[axis_index_from_cartesian["y"]],
        "z": image.shape[axis_index_from_cartesian["z"]]
    }
    z_indices = numpy.linspace( 
        start=0, 
        stop=axis_dimension_from_cartesian["z"]-1, 
        num=num_slices, 
        dtype=int  # forces integer indices
    ) 
    for i in z_indices:
        plot_outline(i)
        plot_slice(i)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=ax_3d, 
        location="left", 
        shrink=0.5, 
        pad=-0.05
    )

    cbar.set_label(r"${q^2}$ (Avg.)", size=11)
    ax_labels = {
        "x": r"$\cos\theta_\mu$",
        "y": r"$\cos\theta_K$",
        "z": r"$\chi$"
    }
    ax_3d.set_xlabel(ax_labels["x"], labelpad=0)
    ax_3d.set_ylabel(ax_labels["y"], labelpad=0)
    ax_3d.set_zlabel(ax_labels["z"], labelpad=-3)
    ax_3d.tick_params(pad=0.3)
    ax_3d.set_box_aspect(None, zoom=0.85)
    note = (
        r"\textbf{Level}: "
        f"{dataset_settings.common.level}\n"
        r"\textbf{Events per set}: "
        f"{dataset_settings.set.num_events_per_set}\n"
        r"\textbf{Bins per dim.}: "
        f"{dataset_settings.num_bins_per_dimension}\n"
        r"$\boldsymbol{\delta C_9}$ : "
        f"{delta_C9_value}"
    )
    ax_3d.set_title(f"{note}", loc="left", y=0.85)
    
    save_plot(dataset_settings=dataset_settings, path_to_plots_dir=path_to_plots_dir)
    plt.close()
    
