
from pathlib import Path

import pandas as pd
import uproot


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