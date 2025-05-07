
import pickle

import torch

from .util import print_gpu_memory_info
from .config import Model_Config
from ..data.dset.dataset import Custom_Dataset
from .models import Custom_Model


class Model_Trainer:
    def __init__(
        self,
        model:Custom_Model, 
        dset:Custom_Dataset,
    ):
        self.model = model
        self.dset = dset

    def save_final(self):
        """
        Save final model.
        """
        
        torch.save(
            self.model.state_dict(), 
            self.model.config.path_file_final
        )

    def save_checkpoint(self, epoch):
        """
        Save checkpoint model.
        """

        path = (
            self.config
            .make_path_file_checkpoint(
                epoch
            )
        )
        torch.save(
            self.model.config.state_dict(), 
            path
        )
    



class Loss_Table:
    def __init__(self):
        self.epochs = []
        self.losses_train = []
        self.losses_eval = []

    def append(
        self, 
        epoch, 
        loss_train, 
        loss_eval
    ):
        self.epochs.append(epoch)
        self.losses_train.append(loss_train)
        self.losses_eval.append(loss_eval)
    
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(
                self, 
                handle, 
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def load(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
            self.epochs = data.epochs
            self.losses_train = data.losses_train
            self.losses_eval = data.losses_eval
            


    


def _train_batch(
    x, 
    y, 
    model, 
    loss_fn, 
    optimizer
):
    """
    Train a model on a single batch 
    given by x, y.
    
    Returns
    -------
    train_loss : float
    """

    model.train()
    
    yhat = model(x)    
    train_loss = loss_fn(yhat, y)

    train_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return train_loss
    

def _evaluate_batch(
    x, 
    y, 
    model, 
    loss_fn
):
    """
    Evaluate model on a mini-batch of data.
    """

    model.eval()
    with torch.no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
        return eval_loss
    

def _train_epoch(
    dataloader, 
    model, 
    loss_fn, 
    optimizer, 
    device=None
):
    """
    Train a model on a dataset.
    """

    num_batches = len(dataloader)
    
    total_batch_loss = 0
    for x, y in dataloader:
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        batch_loss = _train_batch(
            x, 
            y,
            model, 
            loss_fn, 
            optimizer
        )
        total_batch_loss += batch_loss

    avg_batch_loss = (
        total_batch_loss 
        / num_batches
    )

    return avg_batch_loss


def _evaluate_epoch(
    dataloader, 
    model, 
    loss_fn, 
    device=None, 
    scheduler=None
):
    """
    Evaluate a model on a the dataset.
    """
  
    num_batches = len(dataloader)
    
    total_batch_loss = 0
    for x, y in dataloader:
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        batch_loss = _evaluate_batch(
            x, 
            y, 
            model, 
            loss_fn
        )
        total_batch_loss += batch_loss
    
    avg_batch_loss = (
        total_batch_loss 
        / num_batches
    )

    if scheduler:
        scheduler.step(avg_batch_loss)
    
    return avg_batch_loss


def _print_epoch_loss(
    epoch, 
    train_loss, 
    eval_loss
):
    """
    Print a summary of loss values for an epoch.
    """

    print(f"\nEpoch {epoch} complete:")
    print(f"    Train loss: {train_loss}")
    print(f"    Eval loss: {eval_loss}\n")


def _print_prev_learn_rate(scheduler):
    """
    Print the previous learning rate
    given a learning rate scheduler. 
    """

    last_learning_rate = scheduler.get_last_lr()
    message = f"Learning rate: {last_learning_rate}"
    print(message)


def train_and_eval(
    model, 
    train_dataset, 
    eval_dataset,
    loss_fn, 
    optimizer, 
    num_epochs, 
    train_batch_size, 
    eval_batch_size, 
    loss_table:Loss_Table,
    device,
    scheduler=None,
):
    """
    Train and evaluate a model.
    """

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        drop_last=True, 
        shuffle=True
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=eval_batch_size, 
        drop_last=True, 
        shuffle=True
    )
    
    model = model.to(device)

    for ep in range(num_epochs):

        train_loss = _train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimizer, 
            device=device
        ).item()
        
        eval_loss = _evaluate_epoch(
            eval_dataloader, 
            model, 
            loss_fn, 
            device=device, 
            scheduler=scheduler
        ).item()
        
        loss_table.append(
            ep, 
            train_loss, 
            eval_loss
        )
        
        _print_epoch_loss(
            ep, 
            train_loss, 
            eval_loss
        )
        
        if scheduler:
            _print_prev_learn_rate(
                scheduler
            )
        
        print_gpu_memory_info()


        


