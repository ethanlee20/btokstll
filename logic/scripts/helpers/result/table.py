
import torch
import pandas

from ..model.constants import Names_Models
from ..data.dset.constants import (
    Names_Levels,
    Names_q_Squared_Vetos,
    Nums_Events_Per_Set
)

class Summary_Table:

    def __init__(self):

        self.table = self.make_empty()

    def add_item(
        self, 
        level,
        q_squared_veto:str,
        method_name, 
        item_name, 
        num_events_per_set, 
        item,
    ):
        
        if type(item) is torch.Tensor:
            item = item.item()
        
        self.table.loc[
            (
                level, 
                q_squared_veto, 
                method_name, 
                num_events_per_set
            ), 
            item_name,
        ] = item
    
    def reset_table(self):
        
        self.table = self.make_empty()
    
    def make_empty(self):

        index = (
            pandas.MultiIndex
            .from_product(
                Values_Index().tuple_,
                names=Names_Index().tuple_,
            )
        )

        table = pandas.DataFrame(
            index=index, 
            columns=Names_Columns().tuple_
        )
        return table


class Names_Index:

    level = "Level"
    q2_veto = "q^2 Veto"
    method = "Method"
    events_per_set = "Events / Set"

    tuple_ = (
        level,
        q2_veto,
        method,
        events_per_set,
    )


class Values_Index:

    tuple_ = (
        Names_Levels().tuple_, 
        Names_q_Squared_Vetos().tuple_,
        Names_Models().tuple_,
        Nums_Events_Per_Set().tuple_,
    )


class Names_Columns:

    mse = "MSE"
    mae = "MAE"
    np_std = "Std. at NP"
    np_mean = "Mean at NP"
    np_bias = "Bias at NP"

    tuple_ = (
        mse,
        mae,
        np_std,
        np_mean,
        np_bias,
    )