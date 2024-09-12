
"""
A function for generating aggregated datafiles.

Files are aggregated over simulation trials.
"""

import pandas as pd

from ..helpers import file_info, find_raw_data_paths, make_agg_data_filepath

 
def aggregate_raw_data():
    """
    Generate datafiles aggregated over all trials.

    Parameters
    ----------
    split : str
        "train" or "test"
    
    Side Effects
    ------------
    - Save aggregated datafiles 
        (one for the generator level data and
        one for the detector level data).
    """
    
    trial_range = (
        range(1, 31) if split=="train"
        else range(31, 41)
    )

    filepaths = find_raw_data_paths(trial_range)

    df_agg = pd.DataFrame(
        columns=(features + ["dc9"])
    )

    for fp in filepaths:
        
        df_trial = pd.read_pickle(fp)
        
        df_trial = df_trial[features]
        df_trial.dropna(inplace=True)
        
        dc9 = file_info(fp)["dc9"]
        df_trial["dc9"] = dc9

        df_agg = pd.concat(
            [
                None if df_agg.empty else df_agg,
                df_trial
            ]
        )

    for level in {"gen", "det"}:
        df_agg_level = df_agg.loc[level]
        out_filepath = make_agg_data_filepath(level, split)
        df_agg_level.to_pickle(out_filepath)
    


