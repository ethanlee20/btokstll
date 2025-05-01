
import pathlib


def _make_sub_dir_path(dset_name, level, q_squared_veto, dir_path):
    """
    Create the dataset's subdirectory path.
    """
    name = f"{dset_name}_{level}_q2v_{q_squared_veto}"
    path = dir_path.joinpath(name)
    return path


def _make_tensor_filepath(kind, extra_description, split, sub_dir_path):
    """
    Make a filepath for a torch tensor file.

    Parameters
    ----------
    kind : str
        The kind of tensor being saved. e.g. "labels".
        This determines the file name.
    
    Returns
    -------
    path : pathlib.Path
    """
    
    name = (
        f"{extra_description}_{split}_{kind}.pt" 
        if extra_description
        else f"{split}_{kind}.pt"
    )
    path = sub_dir_path.joinpath(name)
    return path


def _convert_split_to_raw_signal_trial_range(split):
    """
    Obtain the raw signal trial range corresponding to
    the data split.
    """
    if split not in ("train", "eval"):
        raise ValueError("Split must be 'train' or 'eval'")
    trial_range = (
        range(1,21) if split=="train"
        else range(21,41) if split=="eval"
        else None
    )
    return trial_range


class Config:
    def __init__(
        self,
        name,
        level,
        q_squared_veto,
        balanced_classes,
        std_scale,
        split,
        dir_path,
        raw_signal_dir_path,
        shuffle=None,
        extra_description=None,
    ):
        self.name = name
        self.level = level
        self.q_squared_veto = q_squared_veto
        self.balanced_classes = balanced_classes
        self.std_scale = std_scale
        self.split = split
        self.dir_path = pathlib.Path(dir_path)
        self.raw_signal_dir_path = raw_signal_dir_path
        self.shuffle = shuffle
        self.extra_description = extra_description
        self.feature_names = [
            "q_squared", 
            "costheta_mu", 
            "costheta_K", 
            "chi"
        ]
        self.label_name = "dc9"
        self.binned_label_name = "dc9_bin_index"
        self.sub_dir_path = _make_sub_dir_path(
            name, 
            level, 
            q_squared_veto, 
            dir_path,
        )
        self.features_path = _make_tensor_filepath(
            "features", 
            extra_description, 
            split, 
            self.sub_dir_path,
        )
        self.labels_path = _make_tensor_filepath(
            "labels",
            extra_description, 
            split, 
            self.sub_dir_path,
        )            
        self.bin_values_path = _make_tensor_filepath(
            "bin_values",       
            extra_description, 
            split, 
            self.sub_dir_path,
        )
        self.raw_signal_trial_range = (
            _convert_split_to_raw_signal_trial_range(split)
        )


                


