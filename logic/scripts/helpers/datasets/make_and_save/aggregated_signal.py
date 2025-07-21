
from pathlib import Path

import numpy
import torch
import pandas

from ..constants import Names_of_Variables, Names_of_Labels


def get_delta_C9_value_of_raw_signal_file(path, verbose=True):

    path = Path(path)
    if not path.is_file():
        raise ValueError
    
    index_of_delta_C9_value_in_split_file_name = 1
    delta_C9_value = float(
        path.name.split('_')
        [index_of_delta_C9_value_in_split_file_name]
    )
    if verbose:
        print(f"Obtained delta C9 value: {delta_C9_value} from file: {path}")
    return delta_C9_value


def get_trial_num_of_raw_signal_file(path, verbose=True):
    
    def convert_string_to_integer(string):
        def assert_string_represents_integer(string):
            assert float(int(string)) == float(string)
        assert_string_represents_integer(string)
        return int(string)
    
    path = Path(path)
    if not path.is_file():
        raise ValueError
    
    index_of_trial_num_in_split_file_name = 2
    trial_num_string = (
        path.name.split('_')
        [index_of_trial_num_in_split_file_name]
    )
    trial_num_int = convert_string_to_integer(trial_num_string)
    if verbose:
        print(f"Obtained trial number: {trial_num_int} from file: {path}")
    return trial_num_int


def make_path_to_aggregated_raw_signal_file(path_to_main_datasets_dir, level, trial_range):
    
    """
    An aggregated raw signal file contains data from
    multiple trials and delta C9 values.
    """
    
    path_to_main_datasets_dir = Path(path_to_main_datasets_dir)
    if not path_to_main_datasets_dir.is_dir():
        raise ValueError
    
    filename = f"agg_sig_{trial_range[0]}_to_{trial_range[-1]}_{level}.pkl"
    path = path_to_main_datasets_dir.joinpath(filename)
    return path


def aggregate_raw_signal_files(
    level, 
    trial_range, 
    list_of_variable_names, 
    path_to_raw_signal_dir, 
    dtype="float64",
    verbose=True,
):
    
    """
    Pickle files only...
    """

    list_of_file_paths = [
        path for path in list(Path(path_to_raw_signal_dir).glob("*.pkl"))
        if get_trial_num_of_raw_signal_file(path, verbose=False) in trial_range
    ]
    list_of_dc9_values = [
        get_delta_C9_value_of_raw_signal_file(path, verbose=verbose) 
        for path in list_of_file_paths
    ]
    list_of_dataframes = [
        pandas.read_pickle(path).loc[level][list_of_variable_names]
        for path in list_of_file_paths
    ]
    list_of_dataframes = [
        dataframe.assign(dc9=dc9) for 
        dataframe, dc9 in zip(list_of_dataframes, list_of_dc9_values)
    ]
    aggregated_dataframe = pandas.concat(list_of_dataframes).astype(dtype)
    if verbose:
        print(
            "Created aggregated signal dataframe.\n"
            f"Trials: {trial_range[0]} to {trial_range[-1]}\n"
            f"Unique delta C9 values: {set(list_of_dc9_values)}"
        )
    return aggregated_dataframe


def add_binned_label_column(
    dataframe, 
    name_of_unbinned_label_column, 
    name_of_binned_label_column
):
    
    """
    Add a binned label column.

    Values are indices of bins.

    Return edited dataframe and a bin map.
    """

    def to_bins(array):
        array = numpy.array(array)
        bin_map, inverse_indices = numpy.unique(
            array, 
            return_inverse=True
        )
        bin_indices = numpy.arange(len(bin_map))
        bins = bin_indices[inverse_indices]
        return bins, bin_map
    
    bins, bin_map = to_bins(
        dataframe[name_of_unbinned_label_column]
    )

    dataframe[name_of_binned_label_column] = bins
    bin_map = torch.from_numpy(bin_map)
    return dataframe, bin_map


class Aggregated_Signal_Dataframe_Handler:

    def __init__(self, path_to_main_datasets_dir, level, trial_range):
        self.level = level
        self.trial_range = trial_range
        self.path_to_dataframe_file = make_path_to_aggregated_raw_signal_file(
            path_to_main_datasets_dir=path_to_main_datasets_dir, 
            level=level, 
            trial_range=trial_range
        )
        self.path_to_bin_map_file = Path(path_to_main_datasets_dir).joinpath("bin_map.pt")

    def get_dataframe(self):
        return pandas.read_pickle(self.path_to_dataframe_file)

    def get_bin_map(self):
        return torch.load(self.path_to_bin_map_file, weights_only=True)

    def make_and_save(self, path_to_raw_signal_dir, dtype="float64", verbose=True):
        dataframe = aggregate_raw_signal_files(
            level=self.level,
            trial_range=self.trial_range,
            list_of_variable_names=Names_of_Variables().tuple_,
            path_to_raw_signal_dir=path_to_raw_signal_dir,
            dtype=dtype,
            verbose=verbose
        )
        dataframe, bin_map = add_binned_label_column(
            dataframe=dataframe, 
            name_of_unbinned_label_column=Names_of_Labels().unbinned, 
            name_of_binned_label_column=Names_of_Labels().binned
        )
        dataframe.to_pickle(self.path_to_dataframe_file)
        torch.save(bin_map, self.path_to_bin_map_file)
