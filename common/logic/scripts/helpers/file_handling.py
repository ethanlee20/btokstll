
import uproot
import pandas
import torch


def open_tree(path_to_root_file, tree_name, verbose=True):
    
    dataframe = (
        uproot.open(f"{path_to_root_file}:{tree_name}")
        .arrays(library="pd")
    )
    if verbose:
        print(f"Opened {path_to_root_file}:{tree_name}")
    return dataframe


def open_root_file_with_one_or_more_trees(path, verbose=True):
    
    """
    Open a root file as a pandas dataframe.

    The file can contain multiple trees.
    Each tree will be labeled by a 
    pandas multi-index.
    """

    def print_message(path, tree_names): 
        def make_message(path, tree_names):
            return f"Opened {path}, containing trees: {', '.join(tree_names)}"
        message = make_message(path=path, tree_names=tree_names)
        print(message)

    file = uproot.open(path)

    tree_names = [
        name.split(';')[0] for name in ("gen", "det")
    ]
    
    list_of_dataframes = [
        file[name].arrays(library="pd") 
        for name in tree_names
    ] 
    final_dataframe = pandas.concat(list_of_dataframes, keys=tree_names)
    if verbose:
        print_message(path=path, tree_names=tree_names)
    return final_dataframe


def save_torch_tensor_as_file(
    tensor, 
    path, 
    verbose=True,
):

    def print_message(tensor, path):
        def make_message(tensor, path):
            return f"Saved tensor of shape: {tensor.shape} to {path}"
        message = make_message(tensor=tensor, path=path)
        print(message)

    torch.save(tensor, path)
    if verbose:
        print_message(tensor=tensor, path=path)


def load_torch_tensor_from_file(
    path,
    verbose=True,  
):
    
    def print_message(tensor, path):
        def make_message(tensor, path):
            return f"Loaded tensor of shape: {tensor.shape} from {path}"
        message = make_message(tensor=tensor, path=path)
        print(message)
    
    tensor = torch.load(path, weights_only=True)
    if verbose: 
        print_message(tensor=tensor, path=path)
    return tensor