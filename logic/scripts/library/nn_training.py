
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
from torch import nn


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


class Custom_Model(nn.Module):
    """Custom model."""
    def __init__(self, nickname, model_dir):
        super().__init__()

        self.nickname = nickname
        self.model_dir = Path(model_dir)
        self.loss_table = self.make_empty_loss_table()
        
    def make_final_save_path(self):
        final_save_path = self.model_dir.joinpath(f"{self.nickname}.pt")
        return final_save_path 
    
    def save_final(self):
        final_save_path = self.make_final_save_path()
        torch.save(self.state_dict(), final_save_path)

    def load_final(self):
        file_path = self.make_final_save_path()
        self.load_state_dict(torch.load(file_path, weights_only=True))
    
    def make_checkpoint_save_path(self, epoch_number):
        checkpoint_save_name = self.nickname + f"_epoch_{epoch_number}"
        checkpoint_save_path = self.model_dir.joinpath(f"{checkpoint_save_name}.pt")
        return checkpoint_save_path
        
    def save_checkpoint(self, epoch_number):
        checkpoint_save_path = self.make_checkpoint_save_path(epoch_number)
        torch.save(self.state_dict(), checkpoint_save_path)

    def load_checkpoint(self, epoch_number):
        file_path = self.make_checkpoint_save_path(epoch_number)
        self.load_state_dict(torch.load(file_path, weights_only=True))

    def make_loss_table_file_path(self):
        file_name = f"{self.nickname}_loss.pkl"
        file_path = self.model_dir.joinpath(file_name)
        return file_path
    
    def save_loss_table(self):
        file_path = self.make_loss_table_file_path()
        with open(file_path, "wb") as handle:
            pickle.dump(self.loss_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_loss_table(self):
        file_path = self.make_loss_table_file_path()
        with open(file_path, "rb") as handle:
            loss_table = pickle.load(handle)
        return loss_table
    
    def append_to_loss_table(self, epoch, train_loss, eval_loss):
        self.loss_table["epoch"].append(epoch)
        self.loss_table["train_loss"].append(train_loss)
        self.loss_table["eval_loss"].append(eval_loss)
        assert len(self.loss_table["epoch"]) == len(self.loss_table["train_loss"]) == len(self.loss_table["eval_loss"])

    def make_empty_loss_table(self):
        """Create an empty loss table."""
        empty_loss_table = {"epoch":[], "train_loss":[], "eval_loss":[]}
        return empty_loss_table
    
    def clear_loss_table(self):
        self.loss_table = self.make_empty_loss_table()


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
    """
    Evaluate model on a mini-batch of data.
    """
    model.eval()
    with torch.no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
        return eval_loss
    

def _train_epoch(dataloader, model, loss_fn, optimizer, data_destination=None):
    """
    Train model over the dataset.
    """
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
    

def _evaluate_epoch(dataloader, model, loss_fn, data_destination=None, scheduler=None):
    """
    Evaluate model over the dataset.
    """
    num_batches = len(dataloader)
    total_batch_loss = 0
    for x, y in dataloader:
        if data_destination is not None:
            x = x.to(data_destination)
            y = y.to(data_destination)
        batch_loss = _evaluate_batch(x, y, model, loss_fn)
        total_batch_loss += batch_loss
    avg_batch_loss = total_batch_loss / num_batches
    if scheduler:
        scheduler.step(avg_batch_loss)
    return avg_batch_loss


def _print_epoch_loss(epoch, train_loss, eval_loss):
    """
    Print a summary of loss values for an epoch.
    """
    print(f"\nepoch {epoch} complete:")
    print(f"    Train loss: {train_loss}")
    print(f"    Eval loss: {eval_loss}\n")


def _print_scheduler_last_learning_rate(scheduler):
    last_learning_rate = scheduler.get_last_lr()
    message = f"learning rate: {last_learning_rate}"
    print(message)


def train_and_eval(
    model:Custom_Model, 
    train_dataset, eval_dataset,
    loss_fn, optimizer, 
    epochs, train_batch_size, eval_batch_size, 
    device, move_data=True,
    scheduler=None,
    checkpoint_epochs=10
):
    """
    Train and evaluate a model.
    """

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True) #, pin_memory=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, drop_last=True, shuffle=True) # , pin_memory=True, num_workers=4)
    
    model = model.to(device)
    data_destination = (device if move_data else None)

    for ep in range(epochs):
        train_loss = _train_epoch(train_dataloader, model, loss_fn, optimizer, data_destination=data_destination).item()
        eval_loss = _evaluate_epoch(eval_dataloader, model, loss_fn, data_destination=data_destination, scheduler=scheduler).item()
        model.append_to_loss_table(ep, train_loss, eval_loss)
        _print_epoch_loss(ep, train_loss, eval_loss)
        if scheduler:
            _print_scheduler_last_learning_rate(scheduler)
        print_gpu_peak_memory_usage()
        if (ep % checkpoint_epochs == 0):
            model.save_checkpoint(ep)

    model.save_final()    
    model.save_loss_table()


