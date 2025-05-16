

import matplotlib.pyplot as plt

from .util import (
    add_plot_note,
    make_note_model,
    save_plot_model
)


def plot_sensitivity(
    preds,
    avg,
    std,
    label,
    config_model,
    path_dir,
    bins=50, 
    xbounds=(-1.5, 0), 
    ybounds=(0, 200), 
):

    _, ax = plt.subplots()

    ax.hist(
        preds, 
        bins=bins, 
        range=xbounds,
    )

    ax.vlines(
        label,
        0,
        ybounds[1],
        color="red",
        label=f"Target ({label})",
    )

    ax.vlines(
        avg,
        0,
        ybounds[1],
        color="red",
        linestyles="--",
        label=(
            r"$\mu = $ " 
            + f"{avg.round(decimals=3)}"
        ),
    )

    height_marker_std = (
        ybounds[1] 
        / 10
    )

    ax.hlines(
        height_marker_std,
        avg,
        avg+std,
        color="orange",
        linestyles="dashdot",
        label=(
            r"$\sigma = $ " 
            + f"{std.round(decimals=3)}"
        ),
    )

    ax.set_xlabel(
        r"Predicted $\delta C_9$"
    )

    ax.set_xbound(*xbounds)
    
    ax.set_ybound(*ybounds)
    
    ax.legend()
    
    note = make_note_model(config_model)

    add_plot_note(
        ax, 
        note, 
        fontsize="medium"
    )

    save_plot_model(
        kind="sens",
        config_model=config_model,
        path_dir=path_dir,
    )

