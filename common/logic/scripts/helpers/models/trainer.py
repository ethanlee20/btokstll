
import torch

from ..plot.loss_curves import plot_loss_curves
from .hardware_util import print_gpu_memory_info
from .loss_table import Loss_Table
from .constants import Names_of_Models
from ..datasets.dataloaders import Custom_Data_Loader


class Trainer:

    def __init__(
        self,
        model, 
        training_dataset,
        evaluation_dataset,
        device,
        start_epoch=0,
    ):    
        self.model = model
        self.training_dataset = training_dataset
        self.dset_eval = evaluation_dataset
        self.device = device
        self.loss_table = self._initialize_loss_table(start_epoch)
        self._make_model_dir()
    
    def train(self):

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.model.settings.training.learning_rate
        )
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            factor=self.model.settings.training.learning_rate_scheduler_reduction_factor, 
            patience=0, 
            threshold=0,
            eps=0
        )

        training_dataloader = (
            Custom_Data_Loader(
                dataset=self.training_dataset, 
                batch_size=self.model.settings.training.size_of_training_batch, 
                drop_last=True, 
                shuffle=True
            )
        )
        evaluation_dataloader = (
            Custom_Data_Loader(
                dataset=self.dset_eval, 
                batch_size=self.model.settings.training.size_of_evaluation_batch, 
                drop_last=True, 
                shuffle=True
            )
        )

        self.model = self.model.to(self.device)
        
        for epoch in range(self.model.settings.training.number_of_epochs):
            training_loss = _train_over_epoch(
                training_dataloader, 
                self.model, 
                self.model.settings.training.loss_fn, 
                optimizer, 
                device=self.device
            ).item()
            evaluation_loss = _evaluate_over_epoch(
                evaluation_dataloader, 
                self.model, 
                self.model.settings.training.loss_fn, 
                device=self.device, 
                scheduler=learning_rate_scheduler
            ).item()
            self.loss_table.append(epoch, training_loss, evaluation_loss)
            _print_epoch_loss(epoch, training_loss, evaluation_loss)
            _print_last_learn_rate(learning_rate_scheduler)
            print_gpu_memory_info()
            if (epoch % self.model.settings.training.number_of_epochs_between_checkpoints) == 0:
                self.model.save_checkpoint_model_to_file(epoch)
                self.loss_table.save(self.model.settings.paths.make_path_to_loss_table_file())
                plot_loss_curves(model_settings=self.model.settings)
                print("Completed checkpoint at epoch: {ep}")
        self.model.save_final_model_to_file()
        self.loss_table.save(self.model.settings.paths.make_path_to_loss_table_file())
        plot_loss_curves(model_settings=self.model.settings)
        
        print("Completed training.")

    def _initialize_loss_table(self, start_epoch):
        if start_epoch == 0:
            loss_table = Loss_Table()
        elif start_epoch > 0:
            loss_table = Loss_Table()
            path_to_existing_loss_table_file  = (
                self.model.settings.paths.make_path_to_loss_table_file()
            )
            loss_table.load(path_to_existing_loss_table_file)
            loaded_reasonable_loss_table = (loss_table.epochs[-1] == start_epoch - 1)
            assert loaded_reasonable_loss_table
        return loss_table

    def _make_model_dir(self): 
        def check_model_dir_does_not_exist(path):
            if path.is_dir():
                raise ValueError(
                    "Model directory already exists. "
                    "Delete directory to retrain."
                )
        path = self.model.settings.paths.path_to_model_dir
        check_model_dir_does_not_exist(path)
        path.mkdir(parents=True, exist_ok=False)


def _train_batch(x, y, model, loss_fn, optimizer):
    
    """
    Train a model on a single batch given by x, y.

    Return training loss.
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
    Evaluate model on a single batch of data.

    Return evaluation loss.
    """
    
    model.eval()
    with torch.no_grad():
        yhat = model(x)
        evaluation_loss = loss_fn(yhat, y)
        return evaluation_loss
    

def _train_over_epoch(dataloader, model, loss_fn, optimizer, device=None):
    
    """
    Train a model on a dataset.
    """
    
    total_batch_loss = 0
    for x, y in dataloader:
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        batch_loss = _train_batch(x, y, model, loss_fn, optimizer)
        total_batch_loss += batch_loss

    num_batches = len(dataloader)
    avg_batch_loss = total_batch_loss / num_batches
    return avg_batch_loss


def _evaluate_over_epoch(dataloader, model, loss_fn, device=None, scheduler=None):
    
    """
    Evaluate a model on a dataset.
    """
    
    total_batch_loss = 0
    for x, y in dataloader:
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        batch_loss = _evaluate_batch(x, y, model, loss_fn)
        total_batch_loss += batch_loss
    
    num_batches = len(dataloader)
    avg_batch_loss = total_batch_loss / num_batches
    if scheduler:
        scheduler.step(avg_batch_loss)
    return avg_batch_loss


def _print_epoch_loss(epoch, train_loss, eval_loss):
    
    print(f"\nEpoch {epoch} complete:")
    print(f"    Train loss: {train_loss}")
    print(f"    Eval loss: {eval_loss}\n")


def _print_last_learn_rate(scheduler):
    
    print(f"Learning rate: {scheduler.get_last_lr()}")
