
from pathlib import Path

from pandas import read_pickle


def make_path_to_raw_bkg_file(
    path_to_raw_bkg_dir,
    charge_or_mix,
    split
):
    
    def make_name_of_raw_bkg_file(charge_or_mix, split):
        if charge_or_mix not in ("charge", "mix"):
            raise ValueError
        name = f"mu_sideb_generic_{charge_or_mix}_{split}.pkl"
        return name
    
    file_name = make_name_of_raw_bkg_file(
        charge_or_mix=charge_or_mix, 
        split=split
    )

    path = Path(path_to_raw_bkg_dir).joinpath(file_name)
    return path


def load_raw_background_file(
    path_to_raw_bkg_dir,
    charge_or_mix,
    split,
    verbose=True,    
):
    
    file_path = make_path_to_raw_bkg_file(
        path_to_raw_bkg_dir=path_to_raw_bkg_dir,
        charge_or_mix=charge_or_mix,
        split=split
    )

    dataframe = read_pickle(file_path)
    if verbose:
        print(f"Loaded raw background file: {file_path}")   
    return dataframe


def load_raw_charge_and_mix_background_files(
    path_to_raw_bkg_dir,
    split,
    verbose=True
):
    
    charge_dataframe = load_raw_background_file(
        path_to_raw_bkg_dir=path_to_raw_bkg_dir,
        charge_or_mix="charge",
        split=split,
        verbose=verbose
    )
    mix_dataframe = load_raw_background_file(
        path_to_raw_bkg_dir=path_to_raw_bkg_dir,
        charge_or_mix="mix",
        split=split,
        verbose=verbose
    )
    return charge_dataframe, mix_dataframe