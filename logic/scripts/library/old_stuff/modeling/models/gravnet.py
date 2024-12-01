
import torch
from torch import nn
from torch_geometric.nn.conv import GravNetConv
from torch_geometric.nn import global_max_pool


class Block(nn.Module):
    """
    A block of the GravNet architecture.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, space_dim, prop_dim, k):
        super().__init__()
        
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gravnet_layer = GravNetConv(
            in_channels=hidden_dim,
            out_channels=output_dim,
            space_dimensions=space_dim,
            propagate_dimensions=prop_dim,
            k=k
        )

    def forward(self, x):
        dense_output = self.dense(x)
        dense_output = dense_output.squeeze()
        gravnet_layer_output = self.gravnet_layer(dense_output)
        return gravnet_layer_output
    

class GravNet_Model(nn.Module):
    """
    The GravNet model.
    """
    
    def __init__(self, input_dim, output_dim, num_blocks, block_hidden_dim, block_output_dim, final_dense_dim, space_dim, prop_dim, k):
        
        super().__init__()

        self.num_blocks = num_blocks
        self.block_output_dim = block_output_dim
        
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                block = Block(input_dim, block_hidden_dim, block_output_dim, space_dim, prop_dim, k)
            elif i != 0:
                block = Block(block_output_dim, block_hidden_dim, block_output_dim, space_dim, prop_dim, k)
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)

        self.dense = nn.Sequential(
            nn.Linear(block_output_dim*num_blocks, final_dense_dim),
            nn.ReLU(),
            nn.Linear(final_dense_dim, output_dim) 
        )
        
    def forward(self, x):
        # breakpoint()
        blocks_output = torch.zeros(x.shape[1], self.block_output_dim*self.num_blocks).to(next(self.parameters()).device)

        for i, block in enumerate(self.blocks):
            x = block(x)
            blocks_output[:, i*self.block_output_dim:(i+1)*self.block_output_dim] = x

        max_output = global_max_pool(blocks_output, batch=None)

        result = self.dense(max_output)

        result = result.squeeze(0)

        return result



