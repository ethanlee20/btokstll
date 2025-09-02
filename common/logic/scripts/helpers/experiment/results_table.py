
import torch
import pandas

from ..models.constants import Names_of_Models
from ..datasets.constants import Names_of_Levels, Names_of_q_Squared_Vetos, Numbers_of_Events_per_Set
from .constants import Names_of_Result_Table_Columns


def make_table_index():

    names = (
        "level",
        "q2_veto",
        "method",
        "events_per_set",
    )

    values = (
        Names_of_Levels().tuple_, 
        Names_of_q_Squared_Vetos().tuple_,
        Names_of_Models().tuple_,
        Numbers_of_Events_per_Set().tuple_,
    )

    index = pandas.MultiIndex.from_product(values, names=names)
    return index


def make_table():

    index = make_table_index()

    table = pandas.DataFrame(
        index=index, 
        columns=Names_of_Result_Table_Columns().tuple_
    )
    return table


class Results_Table:

    def __init__(self):
        self.table = make_table()

    def add_item(self, column_name, value, model_settings, dataset_settings):
        if type(value) is torch.Tensor:
            value = value.item()
        self.table.loc[
            (
                dataset_settings.common.level, 
                dataset_settings.common.preprocessing.q_squared_veto, 
                model_settings.name, 
                dataset_settings.set.num_events_per_set,
            ), 
            column_name,
        ] = value

    def add_items(self, column_names:list, values:list, model_settings, dataset_settings):
        assert type(column_names) == list
        assert type(values) == list
        assert len(column_names) == len(values)
        for column_name, value in zip(column_names, values):
            self.add_item(
                column_name=column_name, 
                value=value, 
                model_settings=model_settings, 
                dataset_settings=dataset_settings
            )

    def reset_table(self):
        self.table = make_table()


