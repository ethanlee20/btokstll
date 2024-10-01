
from torch import nn
import torch


class Phi(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.dense = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        result = self.dense(x)
        return result
    

class Rho(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        result = self.dense(x)
        return result

    

class Deep_Sets(nn.Module):

    def __init__(self):

        super().__init__()

        self.phi = Phi()
        self.rho = Rho()

        self.double()

    def forward(self, x):

        phi_of_x = self.phi(x)
        mean_phi = torch.mean(phi_of_x, 1)
        rho_of_mean_phi = self.rho(mean_phi).squeeze()
        
        return rho_of_mean_phi



