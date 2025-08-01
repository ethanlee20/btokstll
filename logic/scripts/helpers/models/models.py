
import torch

from .constants import Names_of_Models


class Model(torch.nn.Module):

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.file_handler = self.File_Handler(settings)
        self.logic = self._initialize_logic()
        self.double()

    def forward(self, x):
        return self.logic(x)

    def load_final_model_from_file(self):
        state_dict = self.file_handler.loader.load_saved_final_model_state_dict()
        self.logic.load_state_dict(state_dict)

    def load_checkpoint_model_file(self, epoch):
        state_dict = self.file_handler.loader.load_saved_checkpoint_model_state_dict(epoch)
        self.logic.load_state_dict(state_dict)

    def save_final_model_to_file(self):
        state_dict = self.logic.state_dict()
        self.file_handler.saver.save_final_model(state_dict)

    def save_checkpoint_model_to_file(self, epoch):
        state_dict = self.logic.state_dict()
        self.file_handler.saver.save_checkpoint_model(state_dict=state_dict, epoch=epoch)

    def _initialize_logic(self):
        logic = (
            Deep_Sets_Model_Logic()
            if self.settings.name == Names_of_Models().deep_sets
            else CNN_Model_Logic()
            if self.settings.name == Names_of_Models().cnn
            else Event_by_Event_Model_Logic()
            if self.settings.name == Names_of_Models().ebe
            else None
        )
        return logic

    class File_Handler:

        def __init__(self, settings):
            self.loader = self.Model_Loader(settings)
            self.saver = self.Model_Saver(settings)

        class Model_Loader:

            def __init__(self, settings):
                self.settings = settings

            def load_saved_final_model_state_dict(self):
                path = self.settings.paths.make_path_to_final_model_file()
                state_dict = torch.load(path, weights_only=True)
                return state_dict

            def load_saved_checkpoint_model_state_dict(self, epoch):
                path = self.settings.paths.make_path_to_checkpoint_model_file(epoch=epoch)
                state_dict = torch.load(path, weights_only=True)
                return state_dict

        class Model_Saver:

            def __init__(self, settings):
                self.settings = settings

            def save_final_model(self, state_dict):
                path = self.settings.paths.make_path_to_final_model_file()
                torch.save(state_dict, path)

            def save_checkpoint_model(self, state_dict, epoch):
                path = self.settings.paths.make_path_to_checkpoint_model_file(epoch=epoch)
                torch.save(state_dict, path)


class Deep_Sets_Model_Logic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.event_layers = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32)
        )
        self.set_layers = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.event_layers(x)
        x = torch.mean(x, dim=1)
        x = self.set_layers(x)
        x = torch.squeeze(x)
        return x


class CNN_Model_Logic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convolution_layers = torch.nn.Sequential(
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
            self.Residual_Block_A(16, 16),
            self.Residual_Block_A(16, 16),
            self.Residual_Block_A(16, 16),
            self.Residual_Block_B(16, 16),
            self.Residual_Block_A(16, 16),
            self.Residual_Block_A(16, 16),
            self.Residual_Block_A(16, 16),
            self.Residual_Block_B(16, 16),
            self.Residual_Block_A(16, 16),
            self.Residual_Block_A(16, 16),
            self.Residual_Block_A(16, 16)
        )
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        
    def forward(self, x):
        x = self.convolution_layers(x)
        x = torch.mean(x, dim=(2,3,4))
        x = self.dense_layers(x)
        x = torch.squeeze(x)
        return x

    class Residual_Block_A(torch.nn.Module):
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.convolution_block = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=in_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                )
            )
        
        def forward(self, x):
            x = self.convolution_block(x) + x
            x = torch.nn.functional.relu(x)
            return x

    class Residual_Block_B(torch.nn.Module):

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.convolution_block = torch.nn.Sequential(
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
                )
            )
            self.convolution = torch.nn.Conv3d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding="same"
            )
            
        def forward(self, x):
            x = self.convolution(x) + self.convolution_block(x)
            x = torch.nn.functional.relu(x)            
            return x
        

class Event_by_Event_Model_Logic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 44),
        )

    def forward(self, x):
        event_logits = self.layers(x)
        return event_logits



        