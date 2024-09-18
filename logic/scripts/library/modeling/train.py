
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

def _train_batch(x, y, model, loss_fn, optimizer):
    """
    Train a model on a single batch given by x, y.
    
    Returns
    -------
    loss : float
    """
    model.train()
    yhat = model(x)
    train_loss = loss_fn(yhat, y)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return train_loss


def _train_epoch(dataloader, model, loss_fn, optimizer, device):
    num_batches = len(dataloader)
    batch_losses = torch.zeros(num_batches).to(device)

    for batch_index, (x, y) in enumerate(dataloader):
        batch_loss = _train_batch(x, y, model, loss_fn, optimizer)
        batch_losses[batch_index] = batch_loss
    
    epoch_train_loss = torch.mean(batch_losses).item()
    return epoch_train_loss


def _evaluate_batch(x, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
        return eval_loss
    

def _evaluate_epoch(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    batch_losses = torch.zeros(num_batches).to(device)
    
    for batch_index, (x, y) in enumerate(dataloader):
        batch_loss = _evaluate_batch(x, y, model, loss_fn)
        batch_losses[batch_index] = batch_loss
    
    epoch_eval_loss = torch.mean(batch_losses).item()
    return epoch_eval_loss


def _print_epoch_loss(epoch, train_loss, eval_loss):
    print(f"Epoch: {epoch}")
    print(f"Train loss: {train_loss}")
    print(f"Eval loss: {eval_loss}")
    print()


def plot_loss(run_name, epochs, train_losses, eval_losses, ax):
    epochs = list(epochs)
    train_losses = list(train_losses)
    eval_losses = list(eval_losses)

    ax.plot(epochs, train_losses, label="Training Loss")
    ax.plot(epochs, eval_losses, label="Eval. Loss")
    ax.legend()
    ax.set_title(f"{run_name}  (Final eval. loss: {eval_losses[-1]})", loc="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")


def train_and_eval(
    run_name, model, train_dataset, eval_dataset,
    loss_fn, optimizer, device,
    models_dir, plots_dir,
    epochs, train_batch_size, eval_batch_size
):
    models_dir = Path(models_dir)
    plots_dir = Path(plots_dir)

    model_file_name = f"{run_name}.pt"
    model_save_path = models_dir.joinpath(model_file_name)

    loss_plot_file_name = f"loss_{run_name}.png"
    loss_plot_save_path = plots_dir.joinpath(loss_plot_file_name)

    model = model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=True)

    losses = []
    for t in range(epochs):
        train_loss = _train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        eval_loss = _evaluate_epoch(eval_dataloader, model, loss_fn, device)
        epoch_loss = {"epoch":t, "train_loss":train_loss, "eval_loss": eval_loss}
        losses.append(epoch_loss)
        _print_epoch_loss(t, train_loss, eval_loss)

    df_loss = pd.DataFrame.from_records(losses)

    torch.save(model.state_dict(), model_save_path)
    
    # loss plot
    fig, ax = plt.subplots() 
    df_loss = df_loss.iloc[2:]
    plot_loss(
        run_name, 
        df_loss["epoch"], df_loss["train_loss"], df_loss["eval_loss"], 
        ax
    )
    plt.savefig(loss_plot_save_path, bbox_inches="tight")
    plt.close()





