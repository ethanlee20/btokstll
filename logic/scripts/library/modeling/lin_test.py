
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader


def _calculate_linearity(model, eval_dataset, device):
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)
    model = model.to(device)
    model.eval()

    labels = []
    predictions = []
    for x, y in eval_dataloader:
        yhat = model(x)
        predictions.append(yhat.item())
        labels.append(y.item())
    series_result = pd.Series(
        predictions,
        index=labels
    )
    series_result_by_label = series_result.groupby(level=0)
    avgs = series_result_by_label.mean()
    stdevs = series_result_by_label.std()

    np.testing.assert_array_equal(avgs.index, stdevs.index)

    labels = avgs.index
    avgs = avgs.values.tolist()
    stdevs = stdevs.values.tolist()

    return labels, avgs, stdevs 


def _plot_ref_line(start, stop, ax, buffer=0.05, zorder=0):
    """
    Parameters
    ----------
    ax : mpl.axes.Axes
        The axes on which to plot the reference line.
    start_x : float
        The x position of the start of the line.
    stop_x : float
        The x position of the stop of the line.
    buffer : float
        The x-axis distance to enlongate the line beyond the
        start_x and stop_x values. 

    Side Effects
    ------------
    - Plot a reference line on given axes.
    """
    num_ticks = 2
    ticks = np.linspace(start-buffer, stop+buffer, num_ticks)
    ax.plot(
        ticks, ticks,
        label="Ref. Line (Slope = 1)",
        color="grey",
        linewidth=0.5,
        zorder=zorder
    )


def plot_linearity(run_name, model, eval_dataset, device, save_dir):
    
    fig, ax = plt.subplots()
    
    labels, avgs, stdevs = _calculate_linearity(model, eval_dataset, device)
    
    _plot_ref_line(min(labels), max(labels), ax, zorder=0)
    ax.scatter(labels, avgs, label="Validation results", color="firebrick", s=16, zorder=5)
    ax.errorbar(labels, avgs, yerr=stdevs, fmt="none", elinewidth=0.5, capsize=0.5, color="black", label="Std. Dev.", zorder=10)

    ax.set_xlim(-2.25, 1.35)
    ax.set_ylim(-2.25, 1.35)

    ax.legend()
    ax.set_xlabel(r"Actual $\delta C_9$")
    ax.set_ylabel(r"Predicted $\delta C_9$")
    ax.set_title(
        r"\textbf{Linearity} : " + f"{run_name}",
        loc="left"
    )

    save_dir = Path(save_dir)
    file_name = f"lin_{run_name}.png"
    save_path = save_dir.joinpath(file_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


    