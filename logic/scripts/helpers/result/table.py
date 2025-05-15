
import torch
import pandas

from ..model.constants import Names_Models
from ..data.dset.constants import (
    Names_Levels,
    Names_q_Squared_Vetos,
    Nums_Events_Per_Set
)
from ..model.config import Config_Model
from .constants import (
    Info_Index,
    Names_Kinds_Items
)


class Summary_Table:

    def __init__(self):

        self.table = self.make_empty()

    def add_item(
        self, 
        config_model:Config_Model,
        kind,
        item,
    ):

        if kind not in Names_Kinds_Items().tuple_:
            raise ValueError(
                "kind not recognized. "
                f"Must be in: {Names_Kinds_Items().tuple_}"
            )

        if type(item) is torch.Tensor:
            
            item = item.item()

        config_dset = config_model.config_dset
        
        self.table.loc[
            (
                config_dset.level, 
                config_dset.q_squared_veto, 
                config_model.name, 
                config_dset.num_events_per_set,
            ), 
            kind,
        ] = item
    
    def reset_table(self):
        
        self.table = self.make_empty()
    
    def make_empty(self):

        info_index = Info_Index()

        index = (
            pandas.MultiIndex
            .from_product(
                info_index.values.tuple_,
                names=info_index.names.tuple_,
            )
        )

        table = pandas.DataFrame(
            index=index, 
            columns=Names_Kinds_Items().tuple_
        )

        return table


