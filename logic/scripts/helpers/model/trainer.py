

import torch

from .util import print_gpu_memory_info
from .config import Config_Model
from ..data.dset.dataset import Custom_Dataset
from .model import Custom_Model
from .loss_table import Loss_Table


class Model_Trainer:

    """
    Trains a model.
    """

    def __init__(
        self,
        model:Custom_Model, 
        dset_train:Custom_Dataset,
        dset_eval:Custom_Dataset,
        device:str,
    ):
        
        self.model = model

        self.dset_train = dset_train

        self.dset_eval = dset_eval

        self.device = device

        self.loss_table = Loss_Table()

        self._make_dir_model()

    def train(self):

        """
        Train model.
        """

        size_batch_train = (
            self.model.config.size_batch_train
        )
        
        size_batch_eval = (
            self.model.config.size_batch_eval
        )

        loss_fn = self.model.config.fn_loss

        optimizer = self.model.config.optimizer

        lr_scheduler = self.model.config.scheduler_lr

        num_epochs = self.model.config.num_epochs

        num_epochs_checkpoint = (
            self.model.config.num_epochs_checkpoint
        )

        train_dataloader = (
            torch.utils.data.DataLoader(
                self.dset_train, 
                batch_size=size_batch_train, 
                drop_last=True, 
                shuffle=True
            )
        )

        eval_dataloader = (
            torch.utils.data.DataLoader(
                self.dset_eval, 
                batch_size=size_batch_eval, 
                drop_last=True, 
                shuffle=True
            )
        )

        model = model.to(
            self.device
        )

        for ep in range(num_epochs):

            loss_train = _train_epoch(
                train_dataloader, 
                model, 
                loss_fn, 
                optimizer, 
                device=self.device
            ).item()
            
            loss_eval = _evaluate_epoch(
                eval_dataloader, 
                model, 
                loss_fn, 
                device=self.device, 
                scheduler=lr_scheduler
            ).item()
            
            self.loss_table.append(
                ep, 
                loss_train, 
                loss_eval
            )

            _print_epoch_loss(
                ep, 
                loss_train, 
                loss_eval
            )
            
            if lr_scheduler:
                _print_prev_learn_rate(
                    lr_scheduler
                )
            
            print_gpu_memory_info()

            if (ep % num_epochs_checkpoint) == 0:
                self._save_checkpoint(ep)

        self._save_final()

    def _save_final(self):

        """
        Save final.
        """

        path_file_model = (
            self.model.config
            .path_file_final
        )
        
        torch.save(
            self.model.state_dict(), 
            path_file_model,
        )

        self._save_loss_table()

        print("Saved final.")

    def _save_checkpoint(self, epoch):

        """
        Save checkpoint.
        """

        path_file_model = (
            self.model.config
            .make_path_file_checkpoint(
                epoch
            )
        )

        torch.save(
            self.model.state_dict(), 
            path_file_model,
        )

        self._save_loss_table()

        print("Saved checkpoint.")

    def _save_loss_table(self):

        """
        Save the loss table to a file.

        Overwrites file.
        """

        path = (
            self.model.config
            .path_file_loss_table
        )

        self.loss_table.save(
            path
        )

        print("Saved loss table.")

    def _make_dir_model(self): 

        def check_dir_model_not_exist(path):
            
            if path.is_dir():
                raise ValueError(
                    "Model directory already exists. "
                    "Delete directory to retrain."
                )
            
        path = self.model.config.path_dir  

        check_dir_model_not_exist(path)

        path.mkdir(
            parents=True, 
            exist_ok=False,
        )


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




    


        


