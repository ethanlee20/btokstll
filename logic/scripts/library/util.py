
from pathlib import Path

import torch
import pandas as pd
import uproot


## File handling ##

def open_tree(filepath, tree_name):
    """
    Open a root tree as a pandas dataframe.

    Parameters
    ----------
    filepath : str
        Root file's filepath
    tree_name : str
        Tree name

    Returns
    -------
    pd.DataFame
        Root tree dataframe
    """
    df = uproot.open(f"{filepath}:{tree_name}").arrays(library="pd")
    return df


def open_root(filepath):
    """
    Open a root file as a pandas dataframe.

    The file can contain multiple trees.
    Each tree will be labeled by a pandas multi-index.

    Parameters
    ----------
    filepath : str
        Root file's filepath

    Returns
    -------
    pd.DataFrame
        Root file dataframe
    """
    f = uproot.open(filepath)
    tree_names = [name.split(';')[0] for name in f.keys()]
    dfs = [f[name].arrays(library="pd") for name in tree_names] 
    result = pd.concat(dfs, keys=tree_names)
    return result


def open_datafile(filepath):
    """
    Open a datafile as a pandas dataframe.

    The datafile can be a root or pickled pandas dataframe file.
    
    Parameters
    ----------
    filepath : str
        Datafile's filepath
    
    Returns
    -------
    pd.DataFrame
        Datafile dataframe    
    """
    filepath = Path(filepath)
    assert filepath.is_file()
    assert filepath.suffix in {".root", ".pkl"}
    print(f"opening {filepath}")
    if filepath.suffix == ".root":
        return open_root(filepath) 
    elif filepath.suffix == ".pkl":
        return pd.read_pickle(filepath)
    else: raise ValueError("Unknown file type.")


def open_data_dir(dirpath):
    """
    Open all datafiles in a directory (recursively).

    Return a single dataframe containing all the data.

    Parameters
    ----------
    dirpath : str

    Returns
    -------
    pd.DataFrame
        Data dataframe
    """

    dirpath = Path(dirpath)
    assert dirpath.is_dir()
    file_paths = list(dirpath.glob('**/*.root')) + list(dirpath.glob('**/*.pkl'))
    dfs = [open_datafile(path) for path in file_paths]
    if dfs == []:
        raise ValueError("Empty dir.")
    data = pd.concat(dfs)
    return data


def open_data(path):
    """
    Open all datafiles in a directory (if path is a directory).
    Open the specified datafile (if path is a datafile).

    Parameters
    ----------
    path : str
        Path to data directory or datafile
    
    Returns
    -------
    pd.DataFrame
        Data dataframe
    """

    path = Path(path)
    if path.is_file():
        data = open_datafile(path) 
    elif path.is_dir(): 
        data = open_data_dir(path) 

    return data


def get_raw_datafile_info(path):
    path = Path(path)
    dc9 = float(path.name.split('_')[1])
    trial = int(path.name.split('_')[2])
    info = {"dc9": dc9, "trial": trial}
    return info


def aggregate_raw_signal(level, raw_trials:range, columns:list[str], raw_signal_dir_path, dtype="float64"):
    """
    Aggregate data from specified raw signal files.
    
    Returns
    ------
    A dataframe with specified feature columns and a label column named "dc9".
    """
    raw_signal_dir_path = Path(raw_signal_dir_path)

    raw_datafile_paths = []
    for raw_datafile_path in list(raw_signal_dir_path.glob("*.pkl")):
        raw_datafile_trial_number = get_raw_datafile_info(raw_datafile_path)["trial"]
        if raw_datafile_trial_number in raw_trials:
            raw_datafile_paths.append(raw_datafile_path)
    
    num_files_to_load = len(raw_datafile_paths)
    loaded_dataframes = []
    for file_number, path in enumerate(raw_datafile_paths, start=1):
        loaded_dataframe = pd.read_pickle(path).loc[level][columns]
        print(f"opened raw file: [{file_number}/{num_files_to_load}] {path}")
        loaded_dataframes.append(loaded_dataframe)
    loaded_dataframe_dc9_values = [get_raw_datafile_info(path)["dc9"] for path in raw_datafile_paths]
    labeled_dataframe = pd.concat([df.assign(dc9=dc9) for df, dc9 in zip(loaded_dataframes, loaded_dataframe_dc9_values)])
    labeled_dataframe = labeled_dataframe.astype(dtype)
    return labeled_dataframe


## Bootstrapping ##

def bootstrap_labeled_sets(x, y, n, m, reduce_labels=True, labels_to_sample=None):
    """
    Bootstrap m sets of n examples for each label.

    Parameters
    ----------
    x : 
        Array of features of shape: (num_events, num_features)
    y : 
        Array of labels of shape: (num_events)
    n : int
        Number of examples per bootstrap  
    m : int
        Number of bootstraps per label  
    reduce_labels : 
        Whether or not to return one label per set
        (vs one label per example).
    labels_to_sample : list
        Subset of label values to bootstrap.
        If None, bootstrap from all unique label values.

    Returns
    -------
    (bootstrap_x, bootstrap_y) : tuple
        tuple of torch tensors.
    """
    bootstrap_x = []
    bootstrap_y = []
    
    if labels_to_sample is None:
        labels_to_sample = torch.unique(y) 

    for label in labels_to_sample:    

        for iter in range(m):

            pool_x = x[y==label]
            pool_y = y[y==label]
            assert pool_x.shape[0] == pool_y.shape[0]

            selection_indices = torch.randint(low=0, high=len(pool_x), size=(n,))

            label_bootstrap_x = pool_x[selection_indices]
            label_bootstrap_y = pool_y[selection_indices]

            bootstrap_x.append(label_bootstrap_x.unsqueeze(0))
            bootstrap_y.append(label_bootstrap_y.unsqueeze(0))

    bootstrap_x = torch.concatenate(bootstrap_x)
    bootstrap_y = torch.concatenate(bootstrap_y)

    if reduce_labels:
        bootstrap_y = torch.unique_consecutive(bootstrap_y, dim=1).squeeze()
        assert bootstrap_y.shape[0] == bootstrap_x.shape[0]

    return bootstrap_x, bootstrap_y
