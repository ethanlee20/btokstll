

import pathlib

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

from .util import save_plot_dset


def plot_image_slices(
    image, 
    config_dset,
    value_dc9,
    path_dir,
    n_slices=3, 
    cmap=plt.cm.magma, 
):  
    """
    Plot slices of a B->K*ll dataset image.

    Slices are along the chi-axis (axis 2) 
    and might not be evenly spaced.
    """

    fig = plt.figure()

    ax_3d = fig.add_subplot(
        projection="3d"
    )

    cartesian_dim = {
        "x": 0,     
        "y": 1,
        "z": 2,  
    }

    norm = mpl.colors.Normalize(
        vmin=-1, 
        vmax=1,
    )

    image = image.squeeze().cpu()

    colors = cmap(norm(image))
    
    cartesian_shape = {
        "x": image.shape[cartesian_dim["x"]],
        "y": image.shape[cartesian_dim["y"]],
        "z": image.shape[cartesian_dim["z"]],
    }

    def xy_plane(z_pos):

        x, y = numpy.indices(
            (
                cartesian_shape["x"] + 1, 
                cartesian_shape["y"] + 1
            )
        )

        z = numpy.full(
            (
                cartesian_shape["x"] + 1, 
                cartesian_shape["y"] + 1,
            ), z_pos
        )

        return x, y, z
    
    def plot_slice(z_index):

        x, y, z = xy_plane(z_index) 

        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            facecolors=colors[:,:,z_index], 
            shade=False,
        )

    def plot_outline(z_index, offset=0.3):

        x, y, z = xy_plane(
            z_index - offset
        )
        
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, 
            cstride=1, 
            shade=False,
            color="#f2f2f2",
            edgecolor="#f2f2f2", 
        )

    # forces integer indices
    z_indices = numpy.linspace( 
        0, 
        cartesian_shape["z"]-1, 
        n_slices, 
        dtype=int
    ) 
    
    for i in z_indices:

        plot_outline(i)
        plot_slice(i)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=norm, 
            cmap=cmap
        ), 
        ax=ax_3d, 
        location="left", 
        shrink=0.5, 
        pad=-0.05
    )

    cbar.set_label(
        r"${q^2}$ (Avg.)", 
        size=11
    )

    ax_labels = {
        "x": r"$\cos\theta_\mu$",
        "y": r"$\cos\theta_K$",
        "z": r"$\chi$", 
    }

    ax_3d.set_xlabel(
        ax_labels["x"], 
        labelpad=0
    )

    ax_3d.set_ylabel(
        ax_labels["y"], 
        labelpad=0
    )

    ax_3d.set_zlabel(
        ax_labels["z"], 
        labelpad=-3
    )
    
    ax_3d.tick_params(pad=0.3)
    
    ax_3d.set_box_aspect(
        None, 
        zoom=0.85
    )

    note = (
        r"\textbf{Level}: "
        f"{config_dset.level}\n"
        r"\textbf{Events per set}: "
        f"{config_dset.num_events_per_set}\n"
        r"\textbf{Bins per dim.}: "
        f"{config_dset.num_bins_image}\n"
        r"$\boldsymbol{\delta C_9}$ : "
        f"{value_dc9}"
    )

    ax_3d.set_title(
        f"{note}", 
        loc="left",
        y=0.85
    )

    save_plot_dset(
        f"slices_image_dc9_{value_dc9}", 
        config_dset, 
        path_dir,
    )
    
