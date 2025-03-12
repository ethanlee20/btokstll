
from pathlib import Path
import pickle

import torch
from torch import nn


class Custom_Model(nn.Module):
    """Custom model."""
    def __init__(self, name, model_dir, extra_description=None):
        super().__init__()

        self.name = name
        self.model_dir = Path(model_dir)
        self.extra_description = extra_description

        self.base_save_name = (
            f"{self.name}_{self.extra_description}" if extra_description
            else self.name
        )
        self.loss_table = self.make_empty_loss_table()
        
    def make_final_save_path(self):
        final_save_path = self.model_dir.joinpath(f"{self.base_save_name}.pt")
        return final_save_path 
    
    def save_final(self):
        final_save_path = self.make_final_save_path()
        torch.save(self.state_dict(), final_save_path)

    def load_final(self):
        file_path = self.make_final_save_path()
        self.load_state_dict(torch.load(file_path, weights_only=True))
    
    def make_checkpoint_save_path(self, epoch_number):
        checkpoint_save_name = f"{self.base_save_name}_epoch_{epoch_number}"
        checkpoint_save_path = self.model_dir.joinpath(f"{checkpoint_save_name}.pt")
        return checkpoint_save_path
        
    def save_checkpoint(self, epoch_number):
        checkpoint_save_path = self.make_checkpoint_save_path(epoch_number)
        torch.save(self.state_dict(), checkpoint_save_path)

    def load_checkpoint(self, epoch_number):
        file_path = self.make_checkpoint_save_path(epoch_number)
        self.load_state_dict(torch.load(file_path, weights_only=True))

    def make_loss_table_file_path(self):
        file_name = f"{self.base_save_name}_loss.pkl"
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
        assert (
            len(self.loss_table["epoch"]) 
            == len(self.loss_table["train_loss"]) 
            == len(self.loss_table["eval_loss"])
        )

    def make_empty_loss_table(self):
        """Create an empty loss table."""
        empty_loss_table = {"epoch":[], "train_loss":[], "eval_loss":[]}
        return empty_loss_table
    
    def clear_loss_table(self):
        self.loss_table = self.make_empty_loss_table()



##### CNN #####

class CNN_Res(Custom_Model):
    
    def __init__(self, model_dir, extra_description=None):
        super().__init__("cnn_res", model_dir, extra_description=extra_description)

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=1, 
                padding="same", 
                bias=False
            ),
            # nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1),
            *[self.Res_Block(in_out_channels=16) for _ in range(3)],
            self.Conv_Block(in_channels=16, out_channels=16),
            *[self.Res_Block(in_out_channels=16) for _ in range(3)],
            self.Conv_Block(in_channels=16, out_channels=16),
            *[self.Res_Block(in_out_channels=16) for _ in range(3)],
            # Conv_Block(in_channels=128, out_channels=128),
            # *[Res_Block(in_out_channels=128) for _ in range(1)],
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=1),
        )
        
        self.double()

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=(2,3,4))
        x = self.dense(x)
        x = torch.squeeze(x)
        return x

    class Res_Block(nn.Module):
        def __init__(self, in_out_channels):
            super().__init__()
            
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_out_channels, 
                    out_channels=in_out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # nn.BatchNorm3d(num_features=in_out_channels),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=in_out_channels, 
                    out_channels=in_out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # nn.BatchNorm3d(num_features=in_out_channels),
            )
            self.last_activation = nn.ReLU()
        
        def forward(self, x):
            x = self.block(x) + x
            x = self.last_activation(x)
            return x

    class Conv_Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            
            super().__init__()
            
            self.block_a = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # nn.BatchNorm3d(num_features=out_channels),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # nn.BatchNorm3d(num_features=out_channels),
            )
            
            self.block_b = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                # nn.BatchNorm3d(num_features=out_channels),
            )
            
            self.last_activation = nn.ReLU()
        
        def forward(self, x):
            out_block_a = self.block_a(x)
            out_block_b = self.block_b(x)
            x = out_block_a + out_block_b
            x = self.last_activation(x)
            return x






##### Deep Sets #####

class Deep_Sets(Custom_Model):

    def __init__(self, model_dir, extra_description=None):
        super().__init__("deep_sets", model_dir, extra_description=extra_description)

        self.f = nn.Sequential(
            nn.Linear(in_features=4, out_features=32),
            # nn.LayerNorm(normalized_shape=(num_events_per_set, 32)),
            nn.ReLU(),
            *[self.Res_Block_Event(in_out_features=32) for _ in range(3)],
            nn.Linear(in_features=32, out_features=32),
            # nn.LayerNorm(normalized_shape=(num_events_per_set,32)),
        )

        self.g = nn.Sequential(
            *[self.Res_Block_Set(in_out_features=32) for _ in range(3)],
            nn.Linear(in_features=32, out_features=32),
            # nn.BatchNorm1d(num_features=32), 
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )
        
        self.double()

    def forward(self, x):
        x = self.f(x)
        x = torch.mean(x, dim=1)
        x = self.g(x)
        x = torch.squeeze(x)
        return x
    
    class Res_Block_Event(nn.Module):
        def __init__(self, in_out_features):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # nn.LayerNorm(normalized_shape=(num_events_per_set, in_out_features)),
                nn.ReLU(),
                nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # nn.LayerNorm(normalized_shape=(num_events_per_set, in_out_features)),
            )
            self.last_activation = nn.ReLU()
        def forward(self, x):
            x = self.block(x) #+ x
            x = self.last_activation(x)
            return x

    class Res_Block_Set(nn.Module):
        def __init__(self, in_out_features):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # nn.BatchNorm1d(num_features=in_out_features),
                nn.ReLU(),
                nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # nn.BatchNorm1d(num_features=in_out_features),
            )
            self.last_activation = nn.ReLU()
        def forward(self, x):
            x = self.block(x) #+ x
            x = self.last_activation(x)
            return x
    




##### Event-by-event #####

class Event_By_Event_NN(Custom_Model):
    def __init__(self, model_dir, extra_description=None):
        super().__init__("event_by_event_nn", model_dir, extra_description=extra_description)

        self.base = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 44),
        )
        
        self.double()

    def forward(self, x):
        event_logits = self.base(x)
        return event_logits
    
    def predict_log_probabilities(self, x):
        """
        Predict the log probability of each class, given a set of events.

        x : A torch tensor of features of events. (a set)
        """
        with torch.no_grad():
            event_logits = self.forward(x)
            event_log_probabilities = torch.nn.functional.log_softmax(event_logits, dim=1)
            set_logits = torch.sum(event_log_probabilities, dim=0)
            set_log_probabilities = torch.nn.functional.log_softmax(set_logits, dim=0)
        return set_log_probabilities
    
    def calculate_expected_value(self, x, bin_values):
        """
        Calculate the prediction expectation value, given a set of events.

        x : A torch tensor of features of events. (a set)
        """
        with torch.no_grad():
            bin_shift = 5
            bin_values = bin_values + bin_shift
            log_bin_values = torch.log(bin_values)
            log_probs = self.predict_log_probabilities(x)
            lse = torch.logsumexp(log_bin_values + log_probs, dim=0)
            yhat = torch.exp(lse) - bin_shift
        return yhat
    






