
import pathlib

import uproot
import pandas
import torch


def open_tree(path, tree_name, verbose=True):

    """
    Open a tree in a root file as a 
    pandas dataframe.

    Parameters
    ----------
    path : str
        Root file's path.
    tree_name : str
        The name of the tree.
    verbose : bool
        Whether or not to print 
        loading information.

    Returns
    -------
    df : pandas.DataFame
        Dataframe containing the 
        tree's data.
    """

    df = (
        uproot.open(f"{path}:{tree_name}")
        .arrays(library="pd")
    )
    if verbose:
        print(f"Opened {path}:{tree_name}")
    return df


def open_root(path, verbose=True):

    """
    Open a root file as a pandas dataframe.

    The file can contain multiple trees.
    Each tree will be labeled by a 
    pandas multi-index.

    Parameters
    ----------
    path : str
        Root file's path.
    verbose : bool
        Whether or not to print 
        loading information.

    Returns
    -------
    df : pandas.DataFrame
        Root file dataframe
    """

    f = uproot.open(path)
    tree_names = [
        name.split(';')[0] for name in f.keys()
    ]
    dfs = [
        f[name].arrays(library="pd") 
        for name in tree_names
    ] 
    df = pandas.concat(dfs, keys=tree_names)
    if verbose:
        print(
            f"Opened {path}, "
            "containing trees: " 
            f"{", ".join(tree_names)}"
        )
    return df


def open_pickle(path, verbose=True):

    """
    Open a pickled pandas dataframe file.

    Parameters
    ----------
    path : str | pathlib.Path
        The file's path.
    verbose : bool
        Whether or not to print 
        loading information.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing 
        the file's data.
    """

    path = pathlib.Path(path)
    if not path.is_file():
        raise ValueError("Must be a file.")
    if not path.suffix == ".pkl":
        raise ValueError(
            "File must have the '.pkl' suffix."
        )
    df = pandas.read_pickle(path)
    if verbose:
        print(f"Opened {path}")
    return df


def open_data_file(path, verbose=True): 

    """
    Open a data file as a pandas dataframe.

    The data file can be either a root file 
    or a pickled pandas dataframe file.
    
    Parameters
    ----------
    path : str
        Data file's path
    
    Returns
    -------
    df : pandas.DataFrame
        Data file dataframe    
    """

    path = pathlib.Path(path)
    if not path.is_file():
        raise ValueError("Must be a file.")
    
    root_file_suffix = ".root"
    pickle_file_suffix = ".pkl"

    suffix = path.suffix
    if suffix not in {
        root_file_suffix, 
        pickle_file_suffix
    }:
        raise ValueError(
            "File type not readable. "
            "Root and pickle only. "
            "File must have the "
            ".root or .pkl suffix."
        )
    
    read_fn = (
        open_root if suffix == root_file_suffix 
        else open_pickle if suffix == pickle_file_suffix
        else None
    )
    df = read_fn(path, verbose=verbose) 
    
    return df


def open_data_dir(path, verbose=True):

    """
    Open all .pkl and .root data files in a directory 
    (recursively).

    Return a single dataframe containing all the data.

    Parameters
    ----------
    path : str
        The directory's path.
    verbose : bool
        Whether or not to print loading information.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing all data.
    """

    path = pathlib.Path(path)
    if not path.is_dir():
        raise ValueError("Must be a directory.")
    
    file_paths = (
        list(path.glob('**/*.root')) 
        + list(path.glob('**/*.pkl'))
    )
 
    dfs = [
        open_data_file(p, verbose=verbose) 
        for p in file_paths
    ]
    if dfs == []:
        raise ValueError("Empty directory.")
    
    df = pandas.concat(dfs)
    return df


def open_data(path, verbose=True):

    """
    Open all .root and .pkl datafiles 
    in a directory (if path is a directory).

    Open the specified .root or .pkl datafile 
    (if path is a datafile).

    Parameters
    ----------
    path : str
        Path to data directory or datafile
    verbose : bool
        Whether or not to print 
        loading information.
    
    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing all loaded data.
    """

    path = pathlib.Path(path)

    is_dir = path.is_dir()
    is_file = path.is_file()

    if not (is_dir or is_file):
        raise ValueError(
            "Must be a file or directory."
        )

    read_fn = (
        open_data_file if is_file
        else open_data_dir if is_dir
        else None
    )

    df = read_fn(path, verbose=verbose)
    return df


def get_label_file_raw_signal(path, verbose=True):

    """
    Get the label (delta C9 value) 
    of a raw signal data file.
    This information is obtained 
    from the filename.
    
    Parameters
    ----------
    path : str | pathlib.Path
        Path of the raw signal data file.
    verbose : bool
        Whether or not to print 
        file information
        
    Returns
    -------
    dc9 : float
        delta C9 label value.
    """
    
    path = pathlib.Path(path)
    if not path.is_file():
        raise ValueError("Needs to be a file.")
    dc9 = float(path.name.split('_')[1])
    if verbose:
        print(
            f"Obtained label: {dc9} "
            f"from file: {path}"
        )
    return dc9


def get_trial_file_raw_signal(path, verbose=True):

    """
    Get the trial number of a 
    raw signal data file.
    This information is obtained from 
    the filename.

    Parameters
    ----------
    path : str | pathlib.Path
        Path of the raw signal data file.
    verbose : bool
        Whether or not to print 
        file information.

    Returns
    -------
    trial : int
        Trial number
    """
    
    path = pathlib.Path(path)
    if not path.is_file():
        raise ValueError("Needs to be a file.")
    trial = int(path.name.split('_')[2])
    if verbose:
        print(
            f"Obtained trial number: {trial} "
            f"from file: {path}"
        )
    return trial


def make_path_file_agg_raw_signal(
    dir:str, 
    level:str, 
    trials:range
):
    
    """
    Make the save path for an aggregated raw signal data file.
    
    An aggregated raw signal data file contains data from
    multiple raw signal data trial files and from multiple
    delta C9 labels.
    
    Parameters
    ----------
    dir : str
        The path of the directory to 
        save the file to.
    level : str
        The level of simulation. ("gen" or "det")
    trials : range
        The range of trials included in the
        aggregated data file.

    Returns
    -------
    path : pathlib.Path
        The path of the aggregated raw signal data file.
    """
    
    dir = pathlib.Path(dir)
    if not dir.is_dir():
        raise ValueError(
            "Must specify the "
            "path to a directory."
        )

    if not level in {"gen", "det"}:
        raise ValueError("Level must be 'gen' or 'det'.")
    
    filename = f"agg_sig_{trials[0]}_to_{trials[-1]}_{level}.pkl"
    path = dir.joinpath(filename)
    return path


def agg_data_raw_signal(
    level:str, 
    trials:range, 
    columns:list[str], 
    raw_signal_data_dir:str|pathlib.Path, 
    save_dir:str|pathlib.Path=None, 
    dtype:str="float64",
    verbose:bool=True,
):
    
    """
    Aggregate data into a single dataframe 
    from multiple raw signal data files.

    The raw signal data files should be pickled 
    pandas dataframe files.

    Parameters
    ----------
    level : str
        The simulation level.
        ("gen" or "det")
    trials : range
        The range of trial number to include.
    columns : list[str]
        The dataframe columns to include.
    raw_signal_data_dir : str | pathlib.Path
        The source raw signal data directory.
    save_dir : str | pathlib.Path
        The directory to save the resulting
        file to.
        A file will only be saved if this
        is specified.
    dtype : str
        The resulting dataframe's datatype.
    verbose : bool
        Whether or not to print info messages.
    
    Returns
    ------
    df : pandas.DataFrame
        A dataframe with specified feature columns 
        and a label column named "dc9".

    Side Effects
    ------------
    - Save the aggregated data to a file
        (if save_dir is specified).
    """

    raw_signal_data_dir = pathlib.Path(raw_signal_data_dir)
    all_file_paths = list(raw_signal_data_dir.glob("*.pkl"))
    
    selected_file_paths = [
        path for path in all_file_paths 
        if get_trial_file_raw_signal(path, verbose=False) 
        in trials
    ]
    num_files = len(selected_file_paths)
    
    dfs = []
    for i, path in enumerate(selected_file_paths, start=1):
        _df = open_data_file(path, verbose=False).loc[level][columns]
        if verbose: 
            print(f"Opened: [{i}/{num_files}] {path}")
        dfs.append(_df)
    
    dc9_values = [
        get_label_file_raw_signal(path, verbose=False) 
        for path in selected_file_paths
    ]

    df = pandas.concat(
        [_df.assign(dc9=dc9) for _df, dc9 in zip(dfs, dc9_values)]
    )
    df = df.astype(dtype)

    if verbose:
        print(
            "Created aggregated raw signal data file.\n"
            f"Trials: {trials[0]} to {trials[-1]}\n"
            f"Delta C9 values: {set(dc9_values)}"
        )
    
    if save_dir is not None:
        save_path = make_path_file_agg_raw_signal(
            dir=save_dir,
            level=level,
            trials=trials,
        )
        df.to_pickle(save_path)
        if verbose:
            print(f"Saved: {save_path}")

    return df


def load_file_agg_raw_signal(
    dir:str|pathlib.Path, 
    level:str, 
    trials:range, 
    verbose=True
):  
    """
    Load an aggregated raw signal data file as
    a pandas dataframe.

    Parameters
    ----------
    dir : str | pathlib.Path
        The path of the directory
        containing the file.
    level : str
        The reconstruction level of the file.
        ("gen" or "det".)
    trials : range
        The range of trials contained in
        the file.
    verbose : bool
        Whether or not to print information.
    
    Returns
    -------
    df : pandas.DataFrame
        Dataframe of aggregated raw signal data.
    """
    path = make_path_file_agg_raw_signal(
        dir=dir, 
        level=level, 
        trials=trials
    )
    df = open_data_file(path)
    if verbose:
        print(f"Loaded aggregated raw signal data file: {path}")
    return df


def load_file_raw_bkg(
    dir,
    charge_or_mix,
    split,
    verbose=True,    
):
    
    if charge_or_mix not in {"charge", "mix"}:
        raise ValueError("Option not recognized.")
    
    dir = pathlib.Path(dir)

    name = f"mu_sideb_generic_{charge_or_mix}_{split}.pkl"
    
    path = dir.joinpath(name)
    
    df = pandas.read_pickle(path)
    
    if verbose:
        print(f"Loaded raw bkg file: {path}")
        
    return df


def save_file_torch_tensor(
    tensor:torch.Tensor, 
    path:str|pathlib.Path, 
    verbose:bool=True,
):
    """
    Save a torch tensor to a file.
    """

    def print_done_message(tensor, path):
        print(
            f"Generated tensor of shape: "
            f"{tensor.shape}."
            f"\nSaved as: {path}"
        )

    torch.save(tensor, path)    
    if verbose:
        print_done_message(tensor, path)


def load_file_torch_tensor(
    path:str|pathlib.Path,
    verbose:bool=True,  
):
    """
    Load a torch tensor from a file.
    """
    
    def check_file_exists(path):
        if not path.is_file():
            raise ValueError(
                f"File doesn't exist: {path}"
            )
    
    def print_done_message(tensor, path):
        print(
            "Loaded tensor of shape: "
            f"{tensor.shape} "
            f"from: {path}"
        )
    
    check_file_exists(path)
    tensor = torch.load(path, weights_only=True)
    if verbose: 
        print_done_message(tensor, path)
    return tensor