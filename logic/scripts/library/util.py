
from pathlib import Path

import numpy as np
import pandas as pd
import uproot


## Data ##

def bootstrap_over_bins(x, y, n, rng=np.random.default_rng()):
    """
    Bootstrap a set of n items for each unique y.

    x : ndarray of events
    y : ndarray of bins
    n : number of events to sample from each bin    
    rng : numpy random number generator
    """
    bootstrap_x = []
    bootstrap_y = []
    for bin in np.unique(y):
    
        pool_x = x[y==bin]
        pool_y = y[y==bin]
        assert pool_x.shape[0] == pool_y.shape[0]

        selection_indices = rng.choice(len(pool_x), n)

        bin_bootstrap_x = pool_x[selection_indices]
        bin_bootstrap_y = pool_y[selection_indices]

        bootstrap_x.append(bin_bootstrap_x)
        bootstrap_y.append(bin_bootstrap_y)

    bootstrap_x = np.concatenate(bootstrap_x)
    bootstrap_y = np.concatenate(bootstrap_y)

    return bootstrap_x, bootstrap_y


def bootstrap_labeled_sets(df, label, n, m):
    """
    Bootstrap sets from the dataset of each label in a
    source dataframe.

    Bootstrapping refers to sampling with replacement.

    The resulting dataframe has a multi-index where the first
    index is the set index and the second index is the label value.
    
    Parameters
    ----------
    df : pd.DataFrame
        The source dataframe to sample from
    label : str 
        Name of the column specifying the labels.
    n : int
        The number of elements per set.
    m : int
        The number of sets per label. 
    
    Returns
    -------
    sets : pd.DataFrame
        Dataframe of sampled sets.
        Set index is named "set" and label index is named "label".
    """
    df_grouped = df.groupby(label)

    sets = []
    labels = []

    for label_value, df_label in df_grouped:

        for _ in range(m):
            
            df_set = df_label.sample(n=n, replace=True).drop(columns=label)
            sets.append(df_set)
            labels.append(label_value)

    labels = np.array(labels).astype(df[label].dtype)
    set_indices = range(len(sets))
    assert len(set_indices) == len(labels)
    keys = list(zip(set_indices, labels))
    sets = pd.concat(sets, keys=keys, names=["set", "label"])
    return sets



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