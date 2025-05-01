
"""Preprocessing utilities."""

import numpy
import pandas


def _to_bins(ar):

    """
    Translate values in an array to bin numbers.

    Each unique value in the array corresponds to
    a unique bin number.

    Parameters
    ----------
    ar : list | numpy.ndarray | pandas.Series

    Returns
    -------
    bins : numpy.ndarray
        Array of bin numbers.
        A bin number is assigned for each
        value in the original array.
    bin_map : numpy.ndarray
        Map between original values and
        bin numbers.
        Indices of this array correspond
        to bin numbers.
        Values of this array are original values.
    """

    ar = numpy.array(ar)
    bin_map, inverse_indices = numpy.unique(
        ar, 
        return_inverse=True
    )
    bin_indices = numpy.arange(len(bin_map))
    bins = bin_indices[inverse_indices]
    return bins, bin_map


def convert_to_binned(df_agg, label_name, binned_label_name):
    bins, bin_values = _to_bins(df_agg[label_name])
    df_agg[binned_label_name] = bins
    df_agg = df_agg.drop(columns=label_name)
    return df_agg, bin_values


def balance_classes(
    df: pandas.DataFrame, 
    label_column_name: str
):
    
    """
    Reduce the number of events per unique label 
    to the minimum over the labels.

    Shuffles dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.
    label_column_name : str
        The name of the column containing labels.

    Returns
    -------
    df_balanced : pandas.DataFrame
        The balanced dataframe. 
        All data is copied.

    """
    df_shuffled = df.sample(frac=1)
    group_by_label = df_shuffled.groupby(
        label_column_name
    )
    num_events = [
        len(df_label) 
        for _, df_label in group_by_label
    ]
    min_num_events = min(num_events)
    list_df_balanced = [
        df_label[:min_num_events] 
        for _, df_label in group_by_label
    ]
    df_balanced = pandas.concat(list_df_balanced)
    return df_balanced


def _apply_standard_scale():
    pass