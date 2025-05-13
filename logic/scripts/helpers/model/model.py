

import torch

from .config import Config_Model

from .constants import Names_Models


class Custom_Model(torch.nn.Module):
    """
    Custom model.
    """

    def __init__(
        self,
        config:Config_Model,
    ):
        
        super().__init__()

        self.config = config

        if self.config.name == Names_Models().deep_sets:

            self.model = Model_Deep_Sets()

        elif self.config.name == Names_Models().cnn:

            self.model = Model_CNN()

        elif self.config.name == Names_Models().ebe:

            self.model = Model_EBE()

        else:

            raise ValueError(
                f"Name not recognized: "
                f"{self.config.name}"
            )
        
        self.double()
        
    def forward(self, x):

        x = self.model(x)

        return x
    
    def load(self, epoch=None):

        if not epoch:

            path = self.config.path_file_final

        else:

            path = (
                self.config
                .make_path_file_checkpoint(
                    epoch
                )
            )
        
        state_dict = torch.load(
            path, 
            weights_only=True
        )

        self.model.load_state_dict(state_dict)
    

class Model_Deep_Sets(torch.nn.Module):
    
    def __init__(self):

        super().__init__()

        self.f = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            *[self.Block(32) for _ in range(3)],
            torch.nn.Linear(32, 32)
        )

        self.g = torch.nn.Sequential(
            *[self.Block(32) for _ in range(3)],
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):

        x = self.f(x)

        x = torch.mean(x, dim=1)

        x = self.g(x)

        x = torch.squeeze(x)

        return x

    class Block(torch.nn.Module):

        def __init__(self, in_out_feat):

            super().__init__()

            self.block = torch.nn.Sequential(
                torch.nn.Linear(in_out_feat, in_out_feat),
                torch.nn.ReLU(),
                torch.nn.Linear(in_out_feat, in_out_feat),
                torch.nn.ReLU(),
            )

        def forward(self, x):

            x = self.block(x)

            return x
        

class Model_EBE(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.base = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 44),
        )

    def forward(self, x):

        event_logits = self.base(x)

        return event_logits
    
    def predict_log_probs(self, x):

        """
        Predict the log probability 
        of each class, given a set of events.

        x : torch.Tensor 
            Features of events. (a set)
        """

        with torch.no_grad():
            
            event_logits = self.forward(x)
        
            event_log_probs = (
                torch.nn.functional
                .log_softmax(
                    event_logits, 
                    dim=1,
                )
            )

            set_logits = torch.sum(
                event_log_probs, 
                dim=0,
            )
            
            set_log_probs = (
                torch.nn.functional
                .log_softmax(
                    set_logits, 
                    dim=0
                )
            )

        return set_log_probs
    
    def calculate_expected_value(
        log_probs,
        bin_map
    ):
        
        """
        Calculate the prediction expectation 
        value, given log probabilities.

        log_probs : torch.Tensor 
            Log probabilities.
            Output of predict_log_proba_ebe.
        """

        with torch.no_grad():
        
            bin_shift = 5
        
            bin_map = bin_map + bin_shift
        
            log_bin_map = torch.log(bin_map)
        
            lse = torch.logsumexp(
                log_bin_map + log_probs, 
                dim=0
            )
        
            yhat = (
                torch.exp(lse) 
                - bin_shift
            )
        
        return yhat
        

class Model_CNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=1, 
                padding="same", 
                bias=False
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(
                kernel_size=2, 
                stride=1, 
                padding=1
            ),
            *[self.Res_Block(16) for _ in range(3)],
            self.Conv_Block(16, 16),
            *[self.Res_Block(16) for _ in range(3)],
            self.Conv_Block(16, 16),
            *[self.Res_Block(16) for _ in range(3)],
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        
    def forward(self, x):

        x = self.conv(x)

        x = torch.mean(x, dim=(2,3,4))

        x = self.dense(x)

        x = torch.squeeze(x)

        return x

    class Res_Block(torch.nn.Module):

        def __init__(self, in_out_channels):

            super().__init__()
            
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_out_channels, 
                    out_channels=in_out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    in_channels=in_out_channels, 
                    out_channels=in_out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
            )

            self.last_activation = torch.nn.ReLU()
        
        def forward(self, x):

            x = self.block(x) + x

            x = self.last_activation(x)

            return x

    class Conv_Block(torch.nn.Module):

        def __init__(self, in_channels, out_channels):
            
            super().__init__()
            
            self.block_a = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
            )
            
            self.block_b = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
            )
            
            self.last_activation = torch.nn.ReLU()
        
        def forward(self, x):

            out_block_a = self.block_a(x)

            out_block_b = self.block_b(x)

            x = out_block_a + out_block_b

            x = self.last_activation(x)
            
            return x
        