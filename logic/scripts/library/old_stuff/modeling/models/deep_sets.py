
from torch import nn
import torch


class Phi(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.dense = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

    def forward(self, x):
        result = self.dense(x)
        return result
    

class Rho(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.dense = nn.Sequential(
            # nn.Linear(4, 4),
            nn.Linear(4, 1)
        )
    
    def forward(self, x):
        result = self.dense(x)
        return result

    

class Deep_Sets(nn.Module):

    def __init__(self):

        super().__init__()

        self.phi = Phi()
        self.rho = Rho()


    def forward(self, x):
        # breakpoint()
        # phi_of_x = self.phi(x)
        mean_phi = torch.mean(x, 1)
        rho_of_mean_phi = self.rho(mean_phi).squeeze()
        
        return rho_of_mean_phi



