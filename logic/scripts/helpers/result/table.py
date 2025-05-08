
import torch
import pandas


class Summary_Table:
    def __init__(self):
        self.table = self.make_empty()

    def add_item(
        self, 
        level,
        q_squared_veto:bool,
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
        index = pandas.MultiIndex.from_product(
            [
                ["gen", "det"],
                [True, False],
                [
                    "Images", 
                    "Deep Sets", 
                    "Event by event"    
                ],
                [70_000, 24_000, 6_000],
            ],
            names=["Level", "q2_veto", "Method", "Events/set"]
        )
        table = pandas.DataFrame(
            index=index, 
            columns=[
                "MSE",
                "MAE", 
                "Std. at NP", 
                "Mean at NP", 
                "Bias at NP"
            ]
        )
        return table