
import torch
from torch.utils.data import DataLoader


def print_gpu_memory_summary():
    print(torch.cuda.memory_summary(abbreviated=True))


def print_gpu_peak_memory_usage():

    def gpu_peak_memory_usage():
        return f"{torch.cuda.max_memory_allocated()/1024**3:.5f} GB"

    print(f"peak gpu memory usage: {gpu_peak_memory_usage()}")


def select_device():
    """
    Select a device to compute with.

    Returns
    -------
    str
        The name of the selected device.
        "cuda" if cuda is available,
        otherwise "cpu".
    """

    device = (
        "cuda" 
        if torch.cuda.is_available()
        else 
        "cpu"
    )
    print("Device: ", device)
    return device


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


def _evaluate_batch(x, y, model, loss_fn):
    
    model.eval()

    with torch.no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
        return eval_loss


def _train_epoch(dataloader, model, loss_fn, optimizer, data_destination=None):
    num_batches = len(dataloader)
    total_batch_loss = 0
    for x, y in dataloader:
        if data_destination is not None:
            x = x.to(data_destination)
            y = y.to(data_destination)
        batch_loss = _train_batch(x, y, model, loss_fn, optimizer)
        total_batch_loss += batch_loss
    avg_batch_loss = total_batch_loss / num_batches
    return avg_batch_loss
    

def _evaluate_epoch(dataloader, model, loss_fn, data_destination=None):
    num_batches = len(dataloader)
    total_batch_loss = 0
    for x, y in dataloader:
        if data_destination is not None:
            x = x.to(data_destination)
            y = y.to(data_destination)
        batch_loss = _evaluate_batch(x, y, model, loss_fn)
        total_batch_loss += batch_loss
    avg_batch_loss = total_batch_loss / num_batches
    return avg_batch_loss


def _print_epoch_loss(epoch, train_loss, eval_loss):
    print(f"\nepoch {epoch} complete:")
    print(f"    Train loss: {train_loss}")
    print(f"    Eval loss: {eval_loss}\n")


def train_and_eval(
    model, 
    train_dataset, eval_dataset,
    loss_fn, optimizer, 
    epochs, train_batch_size, eval_batch_size, 
    device, move_data=True,
):
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True) #, pin_memory=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, drop_last=True, shuffle=True) #, pin_memory=True, num_workers=4)
    
    model = model.to(device)
    data_destination = (device if move_data else None)

    loss_table = {"epoch":[], "train_loss":[], "eval_loss":[]}
    for ep in range(epochs):
        train_loss = _train_epoch(train_dataloader, model, loss_fn, optimizer, data_destination=data_destination).item()
        eval_loss = _evaluate_epoch(eval_dataloader, model, loss_fn, data_destination=data_destination).item()
        loss_table["epoch"].append(ep)
        loss_table["train_loss"].append(train_loss)
        loss_table["eval_loss"].append(eval_loss)
        _print_epoch_loss(ep, train_loss, eval_loss)
        print_gpu_peak_memory_usage()
    assert len(loss_table["epoch"]) == len(loss_table["train_loss"]) == len(loss_table["eval_loss"])

    return loss_table
