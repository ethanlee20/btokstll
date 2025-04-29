
"""
Functionality for comparing machine learning
approaches for inferring delta C9 from
B -> K* l l events.
"""


import pathlib
import pickle

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import torch
import pandas
import uproot


"""
Datafile handling utilities
"""


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


def get_raw_signal_file_label(path, verbose=True):

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


def get_raw_signal_file_trial(path, verbose=True):

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


"""Basic plotting utilities"""


def setup_high_quality_mpl_params():

    """
    Setup plotting parameters.
    
    Setup environment to make 
    fancy looking plots.
    Inspiration from Chris Ketter.
    Good quality for exporting.

    Side Effects
    ------------
    - Changes matplotlib rc parameters.
    """

    mpl.rcParams["figure.figsize"] = (6, 4)
    mpl.rcParams["figure.dpi"] = 400
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["figure.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["figure.labelsize"] = 30
    mpl.rcParams["xtick.labelsize"] = 12 
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Computer Modern"]
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titley"] = None
    mpl.rcParams["axes.titlepad"] = 4
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["legend.framealpha"] = 0
    mpl.rcParams["legend.markerscale"] = 1
    mpl.rcParams["legend.fontsize"] = 7.5


def make_plot_note(ax, text:str, fontsize="medium"):

    """
    Annotate a plot in the upper right corner,
    above the plot box.

    This doesn't work for 3D plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    text : str
        What to write.
    fontsize : str
        The fontsize.
        Takes any argument that the
        axes.text() method can handle.
    
    Side Effects
    ------------
    - Modifies the given axes.
    """

    ax.text(
        1,
        1.01, 
        text, 
        horizontalalignment="right", 
        verticalalignment="bottom", 
        transform=ax.transAxes, 
        fontsize=fontsize
    )


"""
Math for preprocessing of data.
pandas Dataframe matrix multiplication, etc.
"""


def square_matrix_transform(df_matrix, df_vec):

    """
    Multiply a dataframe of vectors 
    by a dataframe of square matrices.

    Only works for square matrices.
   
    Parameters
    ----------
    df_matrix : pandas.DataFrame
        Dataframe of matrices.
    df_vec : pandas.DataFrame
        Dataframe of vectors.

    Returns
    -------
    result : pandas.DataFrame
        Transformed vector dataframe.
    """

    if not (
        numpy.sqrt(df_matrix.shape[1]) 
        == df_vec.shape[1]
    ):
        raise ValueError("Matrix must be square.")

    dims = df_vec.shape[1]

    result = pandas.DataFrame(
        data=numpy.zeros(shape=df_vec.shape),
        index=df_vec.index,
        columns=df_vec.columns,
        dtype="float64",
    )

    for i in range(dims):
        for j in range(dims):
            result.iloc[:, i] += (
                df_matrix.iloc[:, dims * i + j]
                * df_vec.iloc[:, j]
            )
    return result


def dot_product(df_vec1, df_vec2):
    """
    Compute the dot products of two 
    vector dataframes.
    Computed row-wise.
    
    Parameters
    ----------
    df_vec1 : pandas.DataFrame
        Dataframe of vectors
    df_vec2 : pandas.DataFrame
        Dataframe of vectors
    Returns
    -------
    result : pandas.Series
        A series of the results.
    """

    if not (
        df_vec1.shape[1] == df_vec2.shape[1]
    ):
        raise ValueError(
            "Vector dimensions do not match."
        )
    dims = df_vec1.shape[1]

    result = pandas.Series(
        data=numpy.zeros(len(df_vec1)),
        index=df_vec1.index,
        dtype="float64",
    )
    for dim in range(dims):
        result += (
            df_vec1.iloc[:, dim] 
            * df_vec2.iloc[:, dim]
        )
    return result


def vector_magnitude(df_vec):
    
    """
    Compute the magnitude of each vector 
    in a vector dataframe.

    Parameters
    ----------
    df_vec : pandas.DataFrame
        Dataframe of vectors.

    Returns
    -------
    result : pandas.Series
        Series of the magnitudes.
    """

    result = numpy.sqrt(dot_product(df_vec, df_vec))
    return result


def cosine_angle(df_vec1, df_vec2):
    """
    Find the cosine of the angle 
    between vectors in vector dataframes.
    Computed row-wise.
    
    Parameters
    ----------
    df_vec1 : pandas.DataFrame
        Dataframe of vectors.
    df_vec2 : pandas.DataFrame
        Dataframe of vectors.

    Returns
    -------
    pandas.Series
        A series of the results.
    """

    result = dot_product(df_vec1, df_vec2) / (
        vector_magnitude(df_vec1)
        * vector_magnitude(df_vec2)
    )

    return result


def cross_product_3d(df_3vec1, df_3vec2):

    """
    Find the cross product of vectors of two 
    3-dimensional vector dataframes.
    Computed row-wise.

    Parameters
    ----------
    df_3vec1 : pandas.DataFrame
        Dataframe of 3-dimensional vectors.
    df_3vec2 : pandas.DataFrame
        Dataframe of 3-dimensional vectors.

    Returns
    -------
    result : pandas.DataFrame 
        Dataframe of vectors.
    """

    assert (
        df_3vec1.shape[1] 
        == df_3vec2.shape[1] 
        == 3
    )
    assert df_3vec1.shape[0] == df_3vec2.shape[0]
    assert df_3vec1.index.equals(df_3vec2.index)

    def clean(df_3vec):
        df_3vec = df_3vec.copy()
        df_3vec.columns = ["x", "y", "z"]
        return df_3vec

    df_3vec1 = clean(df_3vec1)
    df_3vec2 = clean(df_3vec2)

    result = pandas.DataFrame(
        data=numpy.zeros(shape=df_3vec1.shape),
        index=df_3vec1.index,
        columns=df_3vec1.columns,
        dtype="float64",
    )

    result["x"] = (
        df_3vec1["y"] * df_3vec2["z"]
        - df_3vec1["z"] * df_3vec2["y"]
    )
    result["y"] = (
        df_3vec1["z"] * df_3vec2["x"]
        - df_3vec1["x"] * df_3vec2["z"]
    )
    result["z"] = (
        df_3vec1["x"] * df_3vec2["y"]
        - df_3vec1["y"] * df_3vec2["x"]
    )

    return result


def unit_normal(df_3vec1, df_3vec2):
    
    """
    Compute the unit normal dataframe of 
    planes specified by two vector dataframes.

    Parameters
    ----------
    df_3vec1 : pandas.DataFrame
        Dataframe of 3 dimensional vectors.
    df_3vec2 : pandas.DataFrame
        Dataframe of 3 dimensional vectors.
    
    Returns
    -------
    df_unit_normal_vec : pandas.DataFrame
        Dataframe of vectors.
    """

    df_normal_vec = cross_product_3d(
        df_3vec1, 
        df_3vec2
    )
    df_unit_normal_vec = df_normal_vec.divide(
        vector_magnitude(df_normal_vec), axis="index"
    )

    return df_unit_normal_vec


"""
Physics calculations for preprocessing data.
Functions for calculating angular 
variables and q^2.
"""


def four_momemtum_dataframe(df_with_4_col):
    
    """
    Create a four-momentum dataframe.

    Create a dataframe where each row 
    represents a four-momentum.
    The columns are well labeled.

    Parameters
    ----------
    df_with_4_col : pandas.DataFrame
        A dataframe with 4 columns.

    Returns
    -------
    df_4mom : pandas.DataFrame
        Well-labeled four-momentum dataframe.
    """

    df_4mom = df_with_4_col.copy()
    df_4mom.columns = ["E", "px", "py", "pz"]
    return df_4mom


def three_momemtum_dataframe(df_with_3_col):

    """
    Create a three-momentum dataframe.

    Create a dataframe where each row 
    represents a three-momentum.
    The columns are well labeled.

    Parameters
    ----------
    df_with_3_col : pandas.DataFrame
        A dataframe with 3 columns.

    Returns
    -------
    df_3mom : pandas.DataFrame
        Well-labeled three-momentum dataframe.
    """

    df_3mom = df_with_3_col.copy()
    df_3mom.columns = ["px", "py", "pz"]
    return df_3mom


def three_velocity_dataframe(df_with_3_col):

    """
    Create a three-velocity dataframe.

    Create a dataframe where each row 
    represents a three-velocity.
    The columns are well labeled.

    Parameters
    ----------
    df_with_3_col : pandas.DataFrame
        A dataframe with 3 columns.

    Returns
    -------
    df_3vel : pandas.DataFrame
        Well-labeled three-velocity dataframe.
    """
    
    df_3vel = df_with_3_col.copy()
    df_3vel.columns = ["vx", "vy", "vz"]
    return df_3vel


def inv_mass_sq_two_particles(
    df_p1_4mom, df_p2_4mom
):
    
    """
    Compute the squares of the invariant masses 
    for two particles systems.

    Parameters
    ----------
    df_p1_4mom : pandas.DataFrame
        Four-momentum dataframe of particle 1
    df_p2_4mom : pandas.DataFrame
        Four momentum dataframe of particle 2

    Returns
    -------
    df_inv_m_sq : pandas.Series
        Series of squared invariant masses.
    """

    df_p1_4mom = four_momemtum_dataframe(df_p1_4mom)
    df_p2_4mom = four_momemtum_dataframe(df_p2_4mom)

    df_sum_4mom = df_p1_4mom + df_p2_4mom
    df_sum_E = df_sum_4mom["E"]
    df_sum_3mom = three_momemtum_dataframe(
        df_sum_4mom[["px", "py", "pz"]]
    )
    df_sum_3mom_mag_sq = (
        vector_magnitude(df_sum_3mom) ** 2
    )

    df_inv_m_sq = df_sum_E**2 - df_sum_3mom_mag_sq
    return df_inv_m_sq


def three_velocity_from_four_momentum_dataframe(df_4mom):

    """
    Compute a three-velocity dataframe 
    from a four-momentum dataframe.
    
    Parameters
    ----------
    df_4mom : pandas.DataFrame
        Dataframe of four-momenta.

    Returns
    -------
    df_3vel : pandas.DataFrame
        Dataframe of three-velocities.
    """

    df_4mom = four_momemtum_dataframe(df_4mom)
    df_relativistic_3mom = df_4mom[["px", "py", "pz"]]
    df_E = df_4mom["E"]
    df_3vel = (
        df_relativistic_3mom.copy()
        .multiply(1 / df_E, axis=0)
        .rename(
            columns={"px": "vx", "py": "vy", "pz": "vz"}
        )
    )
    return df_3vel


def compute_gamma(df_3vel):

    """
    Compute a series of Lorentz factors.

    Parameters
    ----------
    df_3vel : pandas.DataFrame
        Dataframe of three-velocities.

    Returns
    -------
    series_gamma : pandas.Series
        Series of Lorentz factors.
    """

    df_3vel = three_velocity_dataframe(df_3vel)
    series_vel_mag = vector_magnitude(df_3vel)
    series_gamma = 1 / numpy.sqrt(1 - series_vel_mag**2)

    return series_gamma


def compute_Lorentz_boost_matrix(df_3vel):

    """
    Compute a dataframe of Lorentz boost matricies.

    Parameters
    ----------
    df_vel3vec : pandas.DataFrame
        Dataframe of three-velocities.
    
    Returns
    -------
    df_boost_matrix : pandas.DataFrame
        Dataframe of Lorentz boost matricies.
        Each row contains a matrix.
    """

    df_3vel = three_velocity_dataframe(df_3vel)
    df_vel_mag = vector_magnitude(df_3vel)
    df_gamma = compute_gamma(df_3vel)

    df_boost_matrix = pandas.DataFrame(
        data=numpy.zeros(shape=(df_3vel.shape[0], 16)),
        index=df_3vel.index,
        columns=[
            "b00",
            "b01",
            "b02",
            "b03",
            "b10",
            "b11",
            "b12",
            "b13",
            "b20",
            "b21",
            "b22",
            "b23",
            "b30",
            "b31",
            "b32",
            "b33",
        ],
    )

    df_boost_matrix["b00"] = df_gamma
    df_boost_matrix["b01"] = -df_gamma * df_3vel["vx"]
    df_boost_matrix["b02"] = -df_gamma * df_3vel["vy"]
    df_boost_matrix["b03"] = -df_gamma * df_3vel["vz"]
    df_boost_matrix["b10"] = -df_gamma * df_3vel["vx"]
    df_boost_matrix["b11"] = (
        1
        + (df_gamma - 1)
        * df_3vel["vx"] ** 2
        / df_vel_mag**2
    )
    df_boost_matrix["b12"] = (
        (df_gamma - 1)
        * df_3vel["vx"]
        * df_3vel["vy"]
        / df_vel_mag**2
    )
    df_boost_matrix["b13"] = (
        (df_gamma - 1)
        * df_3vel["vx"]
        * df_3vel["vz"]
        / df_vel_mag**2
    )
    df_boost_matrix["b20"] = -df_gamma * df_3vel["vy"]
    df_boost_matrix["b21"] = (
        (df_gamma - 1)
        * df_3vel["vy"]
        * df_3vel["vx"]
        / df_vel_mag**2
    )
    df_boost_matrix["b22"] = (
        1
        + (df_gamma - 1)
        * df_3vel["vy"] ** 2
        / df_vel_mag**2
    )
    df_boost_matrix["b23"] = (
        (df_gamma - 1)
        * df_3vel["vy"]
        * df_3vel["vz"]
        / df_vel_mag**2
    )
    df_boost_matrix["b30"] = -df_gamma * df_3vel["vz"]
    df_boost_matrix["b31"] = (
        (df_gamma - 1)
        * df_3vel["vz"]
        * df_3vel["vx"]
        / df_vel_mag**2
    )
    df_boost_matrix["b32"] = (
        (df_gamma - 1)
        * df_3vel["vz"]
        * df_3vel["vy"]
        / df_vel_mag**2
    )
    df_boost_matrix["b33"] = (
        1
        + (df_gamma - 1)
        * df_3vel["vz"] ** 2
        / df_vel_mag**2
    )

    return df_boost_matrix


def boost(df_ref_4mom, df_4vec):

    """
    Lorentz boost a dataframe of four-vectors.

    Parameters
    ----------
    df_ref_4mom : pandas.DataFrame
        Dataframe of reference four-momenta
        to boost to.
    df_4vec : pandas.DataFrame
        Dataframe of four-vectors to transform.

    Returns
    -------
    df_4vec_transformed : pandas.DataFrame
        Dataframe of boosted four-vectors.
    """

    df_ref_vel = (
        three_velocity_from_four_momentum_dataframe(
            df_ref_4mom
        )
    )
    df_boost_matrix = compute_Lorentz_boost_matrix(
        df_ref_vel
    )
    df_4vec_transformed = square_matrix_transform(
        df_boost_matrix, df_4vec
    )

    return df_4vec_transformed


def find_costheta_ell(
    df_ell_p_4mom, 
    df_ell_m_4mom, 
    df_B_4mom
):
    
    """
    Find the cosine of the muon helicity angle 
    for B -> K* ell+ ell-.

    Parameters
    ----------
    df_ell_p_4mom : pandas.DataFrame
        Dataframe of four-momenta of ell+.
    df_ell_m_4mom : pandas.DataFrame
        Dataframe of four-momenta of ell-.
    df_B_4mom : pandas.DataFrame
        Dataframe of four-momenta of B.
    
    Returns
    -------
    series_costheta_ell : pandas.Series
        Series of cosine muon helicity angles.
    """

    df_ell_p_4mom = four_momemtum_dataframe(df_ell_p_4mom)
    df_ell_m_4mom = four_momemtum_dataframe(df_ell_m_4mom)
    df_B_4mom = four_momemtum_dataframe(df_B_4mom)

    df_ellell_4mom = df_ell_p_4mom + df_ell_m_4mom

    df_ell_p_4mom_ellellframe = boost(
        df_ref_4mom=df_ellell_4mom, df_4vec=df_ell_p_4mom
    )
    df_ell_p_3mom_ellellframe = three_momemtum_dataframe(
        df_ell_p_4mom_ellellframe[["px", "py", "pz"]]
    )

    df_ellell_4mom_Bframe = boost(
        df_ref_4mom=df_B_4mom, df_4vec=df_ellell_4mom
    )
    df_ellell_3mom_Bframe = three_momemtum_dataframe(
        df_ellell_4mom_Bframe[["px", "py", "pz"]]
    )

    series_costheta_ell = cosine_angle(
        df_ellell_3mom_Bframe, df_ell_p_3mom_ellellframe
    )

    return series_costheta_ell


def find_costheta_K(df_K_4mom, df_KST_4mom, df_B_4mom):
    
    """
    Find the cosine of the K* helicity 
    angle for B -> K* ell+ ell-.

    Parameters
    ----------
    df_K_4mom : pandas.DataFrame
        Dataframe of kaon four-momenta.
    df_KST_4mom : pandas.DataFrame
        Dataframe of K* four-momenta.
    df_B_4mom : pandas.DataFrame
        Dataframe of B four-momenta.

    Returns
    -------
    series_costheta_K : pandas.Series
        Series of cosine K* helicity angles.
    """

    df_K_4mom = four_momemtum_dataframe(df_K_4mom)
    df_KST_4mom = four_momemtum_dataframe(df_KST_4mom)
    df_B_4mom = four_momemtum_dataframe(df_B_4mom)

    df_K_4mom_KSTframe = boost(
        df_ref_4mom=df_KST_4mom, df_4vec=df_K_4mom
    )
    df_K_3mom_KSTframe = three_momemtum_dataframe(
        df_K_4mom_KSTframe[["px", "py", "pz"]]
    )

    df_KST_4mom_Bframe = boost(
        df_ref_4mom=df_B_4mom, df_4vec=df_KST_4mom
    )
    df_KST_3mom_Bframe = three_momemtum_dataframe(
        df_KST_4mom_Bframe[["px", "py", "pz"]]
    )

    series_costheta_K = cosine_angle(
        df_KST_3mom_Bframe, df_K_3mom_KSTframe
    )

    return series_costheta_K


def find_unit_normal_KST_K_plane(
    df_B_4mom, df_KST_4mom, df_K_4mom
):
    
    """
    Find the unit normal to the plane made 
    by the direction vectors of the K* and K 
    in B -> K* ell+ ell-.

    Parameters
    ----------
    df_B_4mom : pandas.DataFrame
        Dataframe of the B four-momenta.
    df_KST_4mom : pandas.DataFrame
        Dataframe of the K* four-momenta.
    df_K_4mom : pandas.DataFrame
        Dataframe of the kaon four-momenta.

    Returns
    -------
    df_unit_normal_KST_K_plane : pandas.DataFrame
        Dataframe of unit normal vectors
        from the K* and K plane.
    """

    df_B_4mom = four_momemtum_dataframe(df_B_4mom)
    df_KST_4mom = four_momemtum_dataframe(df_KST_4mom)
    df_K_4mom = four_momemtum_dataframe(df_K_4mom)

    df_K_4mom_KSTframe = boost(
        df_ref_4mom=df_KST_4mom, df_4vec=df_K_4mom
    )
    df_K_3mom_KSTframe = three_momemtum_dataframe(
        df_K_4mom_KSTframe[["px", "py", "pz"]]
    )
    df_KST_4mom_Bframe = boost(
        df_ref_4mom=df_B_4mom, df_4vec=df_KST_4mom
    )
    df_KST_3mom_Bframe = three_momemtum_dataframe(
        df_KST_4mom_Bframe[["px", "py", "pz"]]
    )

    df_unit_normal_KST_K_plane = unit_normal(
        df_K_3mom_KSTframe, df_KST_3mom_Bframe
    )
    return df_unit_normal_KST_K_plane


def find_unit_normal_ellell_ellplus_plane(
    df_B_4mom, df_ell_p_4mom, df_ell_m_4mom
):
    
    """
    Find the unit normal to the plane made by
    the direction vectors of the dilepton system and
    the positively charged lepton in B -> K* ell+ ell-.
    
    Parameters
    ----------
    df_B_4mom : pandas.DataFrame
        Dataframe of B four-momenta.
    df_ell_p_4mom : pandas.DataFrame
        Dataframe of positive lepton four momenta.
    df_ell_m_4mom : pandas.DataFrame
        Dataframe of negative lepton four momenta.

    Returns
    -------
    result : pandas.DataFrame
        Dataframe of unit normal vectors
        from the dilepton and positive lepton plane.
    """

    df_B_4mom = four_momemtum_dataframe(df_B_4mom)
    df_ell_p_4mom = four_momemtum_dataframe(df_ell_p_4mom)
    df_ell_m_4mom = four_momemtum_dataframe(df_ell_m_4mom)

    df_ellell_4mom = df_ell_p_4mom + df_ell_m_4mom

    df_ell_p_4mom_ellellframe = boost(
        df_ref_4mom=df_ellell_4mom, df_4vec=df_ell_p_4mom
    )
    df_ell_p_3mom_ellellframe = three_momemtum_dataframe(
        df_ell_p_4mom_ellellframe[["px", "py", "pz"]]
    )
    df_ellell_4mom_Bframe = boost(
        df_ref_4mom=df_B_4mom, df_4vec=df_ellell_4mom
    )
    df_ellell_3mom_Bframe = three_momemtum_dataframe(
        df_ellell_4mom_Bframe[["px", "py", "pz"]]
    )

    result = unit_normal(
        df_ell_p_3mom_ellellframe, df_ellell_3mom_Bframe
    )
    return result


def find_coschi(
    df_B_4mom,
    df_K_4mom,
    df_KST_4mom,
    df_ell_p_4mom,
    df_ell_m_4mom,
):
    
    """
    Find the cosine of the decay angle chi.

    Chi is the angle between the K* K decay plane 
    and the dilepton ell+ decay plane.

    This is for B -> K* ell+ ell-.

    Parameters
    ----------
    df_B_4mom : pandas.DataFrame
        Dataframe of B four-momenta.
    df_K_4mom : pandas.DataFrame
        Dataframe of kaon four-momenta.
    df_KST_4mom : pandas.DataFrame
        Dataframe of K* four-momenta.
    df_ell_p_4mom : pandas.DataFrame
        Dataframe of positive lepton four-momenta.
    df_ell_m_4mom : pandas.DataFrame
        Dataframe of negative lepton four-momenta.
    
    Returns
    -------
    series_coschi : pandas.Series
        Series of cosine chi values.
    """

    df_unit_normal_KST_K_plane = (
        find_unit_normal_KST_K_plane(
            df_B_4mom, df_KST_4mom, df_K_4mom
        )
    )
    df_unit_normal_ellell_ellplus_plane = (
        find_unit_normal_ellell_ellplus_plane(
            df_B_4mom, df_ell_p_4mom, df_ell_m_4mom
        )
    )

    series_coschi = dot_product(
        df_unit_normal_KST_K_plane,
        df_unit_normal_ellell_ellplus_plane,
    )

    return series_coschi


def find_chi(
    df_B_4mom,
    df_K_4mom,
    df_KST_4mom,
    df_ell_p_4mom,
    df_ell_m_4mom,
):
    """
    Find the decay angle chi.

    Chi is the angle between the K* K decay plane 
    and the dilepton ell+ decay plane.
    It can range from 0 to 2*pi.

    This is for B -> K* ell+ ell-.
    
    Parameters
    ----------
    df_B_4mom : pandas.DataFrame
        Dataframe of B four-momenta.
    df_K_4mom : pandas.DataFrame
        Dataframe of kaon four-momenta.
    df_KST_4mom : pandas.DataFrame
        Dataframe of K* four-momenta.
    df_ell_p_4mom : pandas.DataFrame
        Dataframe of positive lepton four-momenta.
    df_ell_m_4mom : pandas.DataFrame
        Dataframe of negative lepton four-momenta.
    
    Returns
    -------
    series_chi : pandas.Series
        Value of chi for each event.
    """

    series_coschi = find_coschi(
        df_B_4mom,
        df_K_4mom,
        df_KST_4mom,
        df_ell_p_4mom,
        df_ell_m_4mom,
    )

    df_unit_normal_KST_K_plane = (
        find_unit_normal_KST_K_plane(
            df_B_4mom, df_KST_4mom, df_K_4mom
        )
    )
    df_unit_normal_ellell_ellplus_plane = (
        find_unit_normal_ellell_ellplus_plane(
            df_B_4mom, df_ell_p_4mom, df_ell_m_4mom
        )
    )

    df_n_ell_cross_n_K = cross_product_3d(
        df_unit_normal_ellell_ellplus_plane,
        df_unit_normal_KST_K_plane,
    )

    df_B_4mom = four_momemtum_dataframe(df_B_4mom)
    df_KST_4mom = four_momemtum_dataframe(df_KST_4mom)
    df_KST_4mom_Bframe = boost(
        df_ref_4mom=df_B_4mom, 
        df_4vec=df_KST_4mom
    )
    df_KST_3mom_Bframe = three_momemtum_dataframe(
        df_KST_4mom_Bframe[["px", "py", "pz"]]
    )

    series_n_ell_cross_n_K_dot_Kst = dot_product(
        df_n_ell_cross_n_K, df_KST_3mom_Bframe
    )
    series_chi = (
        numpy.sign(series_n_ell_cross_n_K_dot_Kst) 
        * numpy.arccos(series_coschi)
    )

    def to_positive_angles(chi):
        return chi.where(chi > 0, chi + 2 * numpy.pi)

    series_chi = to_positive_angles(series_chi)

    return series_chi


def calc_dif_inv_mass_k_pi_and_kst(df_K_4mom, df_pi_4mom):

    """
    Calcualate the difference between the 
    invariant mass of the K pi system
    and the K*'s invariant mass (PDG value).

    Parameters
    ----------
    df_K_4mom : pandas.DataFrame
        Dataframe of kaon four-momenta.

    df_pi_4mom : pandas.DataFrame
        Dataframe of pion four-momenta.

    Returns
    -------
    series_dif : pandas.Series
        Series of differences.
    """

    inv_mass_kst = 0.892

    df_inv_mass_k_pi = numpy.sqrt(
        inv_mass_sq_two_particles(
            df_K_4mom, 
            df_pi_4mom
        )
    )

    series_dif = df_inv_mass_k_pi - inv_mass_kst
    return series_dif


def calculate_variables(ell, df):

    """
    Calculate decay variables 
    from B -> K* ell+ ell- data.

    Parameters
    ----------
    ell : str
        "mu" or "e"
    df : pandas.DataFrame
        Dataframe of event data.

    Returns
    -------
    df_result : pandas.DataFrame
        Dataframe of event data with 
        additional calculated variables.
    """

    if ell not in {"mu", "e"}:
        raise ValueError(f"ell not recognized: {ell}")

    ell_p_E = f'{ell}_p_E'
    ell_p_px = f'{ell}_p_px'
    ell_p_py = f'{ell}_p_py'
    ell_p_pz = f'{ell}_p_pz'
    ell_p_mcE = f'{ell}_p_mcE'
    ell_p_mcPX = f'{ell}_p_mcPX'
    ell_p_mcPY = f'{ell}_p_mcPY'
    ell_p_mcPZ = f'{ell}_p_mcPZ'
    ell_m_E = f'{ell}_m_E'
    ell_m_px = f'{ell}_m_px'
    ell_m_py = f'{ell}_m_py'
    ell_m_pz = f'{ell}_m_pz'
    ell_m_mcE = f'{ell}_m_mcE'
    ell_m_mcPX = f'{ell}_m_mcPX'
    ell_m_mcPY = f'{ell}_m_mcPY'
    ell_m_mcPZ = f'{ell}_m_mcPZ'
    costheta_ell = f'costheta_{ell}'
    costheta_ell_mc = f'costheta_{ell}_mc'

    df_B_4mom = four_momemtum_dataframe(
        df[["E", "px", "py", "pz"]]
    )
    df_B_4mom_mc = four_momemtum_dataframe(
        df[["mcE", "mcPX", "mcPY", "mcPZ"]]
    )
    df_ell_p_4mom = four_momemtum_dataframe(
        df[[ell_p_E, ell_p_px, ell_p_py, ell_p_pz]]
    )
    df_ell_p_4mom_mc = four_momemtum_dataframe(
        df[[ell_p_mcE, ell_p_mcPX, ell_p_mcPY, ell_p_mcPZ]]
    )
    df_ell_m_4mom = four_momemtum_dataframe(
        df[[ell_m_E, ell_m_px, ell_m_py, ell_m_pz]]
    )
    df_ell_m_4mom_mc = four_momemtum_dataframe(
        df[[ell_m_mcE, ell_m_mcPX, ell_m_mcPY, ell_m_mcPZ]]
    )
    df_K_4mom = four_momemtum_dataframe(
        df[["K_p_E", "K_p_px", "K_p_py", "K_p_pz"]]
    )
    df_K_4mom_mc = four_momemtum_dataframe(
        df[["K_p_mcE", "K_p_mcPX", "K_p_mcPY", "K_p_mcPZ"]]
    )
    df_pi_4mom = four_momemtum_dataframe(
        df[["pi_m_E", "pi_m_px", "pi_m_py", "pi_m_pz"]]
    )
    df_pi_4mom_mc = four_momemtum_dataframe(
        df[["pi_m_mcE", "pi_m_mcPX", "pi_m_mcPY", "pi_m_mcPZ"]]
    )
    df_KST_4mom = four_momemtum_dataframe(
        df[["KST0_E", "KST0_px", "KST0_py", "KST0_pz"]]
    )
    df_KST_4mom_mc = four_momemtum_dataframe(
        df[["KST0_mcE", "KST0_mcPX", "KST0_mcPY", "KST0_mcPZ"]]
    )

    df_result = df.copy()

    df_result["q_squared"] = inv_mass_sq_two_particles(
        df_ell_p_4mom, df_ell_m_4mom
    )
    df_result["q_squared_mc"] = inv_mass_sq_two_particles(
        df_ell_p_4mom_mc, df_ell_m_4mom_mc
    )
    df_result[costheta_ell] = find_costheta_ell(
        df_ell_p_4mom, df_ell_m_4mom, df_B_4mom
    )
    df_result[costheta_ell_mc] = find_costheta_ell(
        df_ell_p_4mom_mc, df_ell_m_4mom_mc, df_B_4mom_mc
    )
    df_result["costheta_K"] = find_costheta_K(
        df_K_4mom, df_KST_4mom, df_B_4mom
    )
    df_result["costheta_K_mc"] = find_costheta_K(
        df_K_4mom_mc, df_KST_4mom_mc, df_B_4mom_mc
    )
    df_result["coschi"] = find_coschi(
        df_B_4mom,
        df_K_4mom,
        df_KST_4mom,
        df_ell_p_4mom,
        df_ell_m_4mom,
    )
    df_result["coschi_mc"] = find_coschi(
        df_B_4mom_mc,
        df_K_4mom_mc,
        df_KST_4mom_mc,
        df_ell_p_4mom_mc,
        df_ell_m_4mom_mc,
    )
    df_result["chi"] = find_chi(
        df_B_4mom,
        df_K_4mom,
        df_KST_4mom,
        df_ell_p_4mom,
        df_ell_m_4mom,
    )
    df_result["chi_mc"] = find_chi(
        df_B_4mom_mc,
        df_K_4mom_mc,
        df_KST_4mom_mc,
        df_ell_p_4mom_mc,
        df_ell_m_4mom_mc,
    )
    df_result["invM_K_pi_shifted"] = calc_dif_inv_mass_k_pi_and_kst(
        df_K_4mom,
        df_pi_4mom
    )
    df_result["invM_K_pi_shifted_mc"] = calc_dif_inv_mass_k_pi_and_kst(
        df_K_4mom_mc,
        df_pi_4mom_mc
    )

    return df_result


"""
Further preprocessing utilities
"""


def make_aggregated_raw_signal_file_save_path(
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


def aggregate_raw_signal_data_files(
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
        if get_raw_signal_file_trial(path, verbose=False) 
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
        get_raw_signal_file_label(path, verbose=False) 
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
        save_path = make_aggregated_raw_signal_file_save_path(
            dir=save_dir,
            level=level,
            trials=trials,
        )
        df.to_pickle(save_path)
        if verbose:
            print(f"Saved: {save_path}")

    return df


def load_aggregated_raw_signal_data_file(
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
    path = make_aggregated_raw_signal_file_save_path(
        dir=dir, 
        level=level, 
        trials=trials
    )
    df = open_data_file(path)
    if verbose:
        print(f"Loaded aggregated raw signal data file: {path}")
    return df


def bootstrap_labeled_sets(
    x:torch.Tensor, 
    y:torch.Tensor, 
    n:int, 
    m:int, 
    reduce_labels:bool=True, 
    labels_to_sample:list[float]=None
):
    
    """
    Bootstrap m sets of n examples for each unique label.

    Parameters
    ----------
    x : torch.Tensor
        Array of features of shape: 
        (number of events, number of features)
    y : torch.Tensor
        Array of labels of shape: (number of events)
    n : int
        Number of events per bootstrap  
    m : int
        Number of bootstraps per unique label  
    reduce_labels : bool
        Whether or not to return one label per bootstrap
        (versus one label per event).
    labels_to_sample : list
        Subset of label values to bootstrap.
        If None, bootstrap from all unique label values.

    Returns
    -------
    bootstrap_x : torch.Tensor
        Torch tensor of bootstrapped features.
        Shape (m, n, number of features).
    bootstrap_y : torch.Tensor
        Torch tenor of bootstrapped labels.
        Shape (m) if reduce_labels is True.
        Shape (m, n) if reduce_labels is False.
    """

    bootstrap_x = []
    bootstrap_y = []
    
    if labels_to_sample is None:
        labels_to_sample = torch.unique(
            y, 
            sorted=True
        ) 
    else:
        labels_to_sample, labels_to_sample_counts = (
            torch.unique(
                labels_to_sample, 
                sorted=True,
                return_counts=True,
            )
        )
        if torch.any(labels_to_sample_counts > 1):
            raise ValueError(
                "Cannot bootstrap from subset " \
                "containing repeated labels."
            )

    for label in labels_to_sample:    

        for _ in range(m):

            pool_x = x[y==label]
            pool_y = y[y==label]
            assert pool_x.shape[0] == pool_y.shape[0]

            selection_indices = torch.randint(
                low=0, 
                high=len(pool_x), 
                size=(n,)
            )

            _bootstrap_x = pool_x[selection_indices]
            _bootstrap_y = pool_y[selection_indices]

            bootstrap_x.append(_bootstrap_x.unsqueeze(0))
            bootstrap_y.append(_bootstrap_y.unsqueeze(0))

    bootstrap_x = torch.concatenate(bootstrap_x)
    bootstrap_y = torch.concatenate(bootstrap_y)

    if reduce_labels:
        bootstrap_y = torch.unique_consecutive(bootstrap_y, dim=1).squeeze()
        assert bootstrap_y.shape[0] == bootstrap_x.shape[0]

    return bootstrap_x, bootstrap_y


def drop_na(df, verbose=True):

    """
    Drop rows of a dataframe that contain a NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.

    Returns
    -------
    df_out : pandas.DataFrame
        A modified copy of the dataframe.
    """

    if verbose:
        print("Number of NA values: ", df.isna().sum())
    df_out = df.dropna()
    if verbose:
        print("Removed NA rows.")
    return df_out


def apply_q_squared_veto(df: pandas.DataFrame, strength:str):
    
    """
    Apply a q^2 veto to a dataframe of B->K*ll events.

    Parameters
    ---------- 
    df : pandas.DataFrame
        The dataframe of events.
    strength : str
        The strength of the veto. ('tight' or 'loose') 
        Tight keeps  1 < q^2 < 8.
        Loose keeps 0 < q^2 < 20.

    Returns
    -------
    df_vetoed : pandas.DataFrame
        The reduced dataframe.
        Returns a copy.
    """

    if strength not in ("tight", "loose"):
        raise ValueError(
            "strength must be 'tight' or 'loose'"
        )

    tight_bounds = (1, 8) 
    loose_bounds = (0, 20)

    bounds = (
        tight_bounds if strength == "tight"
        else loose_bounds if strength == "loose"
        else None
    )

    lower_bound = bounds[0]
    upper_bound = bounds[1]

    df_vetoed = df[
        (df["q_squared"]>lower_bound) 
        & (df["q_squared"]<upper_bound)
    ].copy()
    return df_vetoed


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


def get_dataset_prescale(
    kind:str, 
    level:str, 
    q_squared_veto:str, 
    var:str,
):
    """
    Get prescale values calculated from the
    first 20 trials (training dataset).

    Prescale values are used for standard scaling
    the dataset's features.

    Parameters
    ----------
    kind : str
        The kind of value to get.
        "mean" or "std".
    level : str
        The reconstruction level.
        "gen" or "det".
        Determines which dataset's values to get.
    q_squared_veto : str
        "tight" or "loose". "tight" gets values computed
        from the dataset with a tight q^2 veto.
        "loose" gets values computed from the dataset
        with a loose q^2 veto.
    var : str
        The variable whose value to get.
        "q_squared", "costheta_mu", "costheta_K" 
        or "chi"

    Returns 
    -------
    value : float
        The requested prescale value.

    """

    table_tight_q2_veto = {
        "mean": {
            "gen": {
                "q_squared": 4.752486,
                "costheta_mu": 0.065137,
                "costheta_K": 0.000146,
                "chi": 3.141082, 
            }, 
            "det": {
                "q_squared": 4.943439,
                "costheta_mu": 0.077301,
                "costheta_K": -0.068484,
                "chi": 3.141730, 
            }
        },
        "std": {
            "gen": {
                "q_squared": 2.053569,
                "costheta_mu": 0.505880,
                "costheta_K": 0.694362,
                "chi": 1.811370, 
            }, 
            "det": {
                "q_squared": 2.030002,
                "costheta_mu": 0.463005,
                "costheta_K": 0.696061,
                "chi": 1.830277, 
            }
        }
    }

    table_loose_q2_veto = {
        "mean": {
            "gen": {
                "q_squared": 9.248781,
                "costheta_mu": 0.151290,
                "costheta_K": 0.000234,
                "chi": 3.141442, 
            }, 
            "det": {
                "q_squared": 10.353162,
                "costheta_mu": 0.177404,
                "costheta_K": -0.031136,
                "chi": 3.141597, 
            }
        },
        "std": {
            "gen": {
                "q_squared": 5.311177,
                "costheta_mu": 0.524446,
                "costheta_K": 0.635314,
                "chi": 1.803100, 
            }, 
            "det": {
                "q_squared": 5.242896,
                "costheta_mu": 0.508787,
                "costheta_K": 0.622743,
                "chi": 1.820018, 
            }    
        }
    }

    table = (
        table_tight_q2_veto if q_squared_veto == "tight"
        else table_loose_q2_veto if q_squared_veto == "loose"
        else None
    )

    value = table[kind][level][var]
    return value


def to_bins(ar):

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


def make_image(ar_feat, n_bins):

    """
    Make an image of a B->K*ll dataset. 
    Like Shawn.

    Order of input features must be:             
        "q_squared", 
        "costheta_mu", 
        "costheta_K", 
        "chi"

    Parameters
    ----------
    ar_feat : array_like
        Array of features.
    n_bins : int
        Number of bins per dimension.

    Returns
    -------
    image : torch.Tensor
        Torch tensor of the average q^2 calculated 
        per 3D angular bin.
        Shape of (1, n_bins, n_bins, n_bins).
        A 3D image with 1 color channel.
        Empty bins set to 0.
    """

    angular_features = ar_feat[:,1:]
    q_squared_features = ar_feat[:,0]
    np_image = scipy.stats.binned_statistic_dd(
        sample=angular_features,
        values=q_squared_features, 
        bins=n_bins,
        range=[(-1, 1), (-1, 1), (0, 2*numpy.pi)]
    ).statistic
    torch_image = torch.from_numpy(np_image)
    torch_image = torch.nan_to_num(torch_image)
    torch_image = torch_image.unsqueeze(0)
    return torch_image


def get_num_per_unique_label(labels:torch.Tensor):

    """
    Get the number of examples with each unique label.

    Only works if there is the same number
    of examples per each unique label.

    Parameters
    ----------
    labels : torch.Tensor
        Array of labels. One label per example.
    
    Returns
    -------
    num_per_label : int
        Number of examples of each unique label.
    """

    _, label_counts = torch.unique(
        labels, 
        return_counts=True
    )
    # check same number of sets per label
    assert torch.all(label_counts == label_counts[0]) 
    num_per_label = label_counts[0].item()
    return num_per_label


def plot_image_slices(
    image, 
    n_slices=3, 
    cmap=plt.cm.magma, 
    note="",
):
    """
    Plot slices of a B->K*ll dataset image.

    Slices are along the chi-axis (axis 2) and might
    not be evenly spaced.

    Parameters
    ----------
    image : torch.Tensor
        Tensor created by the make_image function.
        Dimensions must correspond to 
        (costheta_mu, costheta_K, chi).
    n_slices : int
        The number of slices to show.
    cmap : matplotlib.colors.Colormap
        The colormap.
    note : str
        Add an annotation to the plot.
    
    Side Effects
    ------------
    - Creates a plot.
    """

    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d")

    var_dim = {
        0: "costheta_mu",
        1: "costheta_K",
        2: "chi",
    }

    cartesian_dim = {
        "x": 0,     
        "y": 1,
        "z": 2,  
    }

    norm = mpl.colors.Normalize(vmin=-1.1, vmax=1.1)
    image = image.squeeze().cpu()
    colors = cmap(norm(image))
    
    cartesian_shape = {
        "x": image.shape[cartesian_dim["x"]],
        "y": image.shape[cartesian_dim["y"]],
        "z": image.shape[cartesian_dim["z"]],
    }

    def xy_plane(z_pos):
        x, y = numpy.indices(
            (
                cartesian_shape["x"] + 1, 
                cartesian_shape["y"] + 1
            )
        )
        z = numpy.full(
            (
                cartesian_shape["x"] + 1, 
                cartesian_shape["y"] + 1,
            ), z_pos
        )
        return x, y, z
    
    def plot_slice(z_index):
        x, y, z = xy_plane(z_index) 
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            facecolors=colors[:,:,z_index], 
            shade=False,
        )

    def plot_outline(z_index, offset=0.3):
        x, y, z = xy_plane(
            z_index - offset
        )
        
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, 
            cstride=1, 
            shade=False,
            color="#f2f2f2",
            edgecolor="#f2f2f2", 
        )

    z_indices = numpy.linspace( # forces integer indices
        0, 
        cartesian_shape["z"]-1, 
        n_slices, 
        dtype=int
    ) 
    
    for i in z_indices:
        plot_outline(i)
        plot_slice(i)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=ax_3d, 
        location="left", 
        shrink=0.5, 
        pad=-0.05
    )
    cbar.set_label(r"${q^2}$ (Avg.)", size=11)

    ax_labels = {
        "x": r"$\cos\theta_\mu$",
        "y": r"$\cos\theta_K$",
        "z": r"$\chi$", 
    }
    ax_3d.set_xlabel(ax_labels["x"], labelpad=0)
    ax_3d.set_ylabel(ax_labels["y"], labelpad=0)
    # ax_3d.zaxis.set_rotate_label(False)
    ax_3d.set_zlabel(ax_labels["z"], labelpad=-3)

    ticks = {
        "x": ["-1", "1"],
        "y": ["-1", "1"],
        "z": ['0', r"$2\pi$"],
    }      

    ax_3d.set_xticks([0, cartesian_shape["x"]-1], ticks["x"])
    ax_3d.set_xticks([0, cartesian_shape["y"]-1], ticks["y"])
    ax_3d.set_xticks([0, cartesian_shape["z"]-1], ticks["z"])
    ax_3d.tick_params(pad=0.3)
    ax_3d.set_box_aspect(None, zoom=0.85)
    ax_3d.set_title(f"{note}", loc="center", y=1)


"""
Datasets.
pytorch datasets for training / evaluating models.
"""


class Custom_Dataset(torch.utils.data.Dataset):
    """
    Custom dataset base class.
    """
    def __init__(
        self, 
        name, 
        level, 
        q_squared_veto, 
        split, 
        save_dir, 
        extra_description=None, 
        regenerate=False
    ):
        """
        Parameters
        ----------
        name : str
            The name of the dataset.
            e.g. "images"
        level : str
            The reconstruction level.
            Either "gen" or "det".
        q_squared_veto : bool
            Whether or not to apply a veto
            in q^2.
        split : str
            The dataset split.
            Either "train" or "eval".
        save_dir : str | pathlib.Path
            The directory to save the dataset to.
        extra_description : str
            Any extra information to identify
            the dataset.
        regenerate : bool
            Whether or not to regenerate (and save)
            the dataset.
        """
        self.name = name 
        self.extra_description = extra_description
        self.level = level
        self.q_squared_veto = q_squared_veto
        self.split = split
        self.save_dir = pathlib.Path(save_dir)

        save_sub_dir_name = f"{name}_{level}_q2v_{q_squared_veto}"
        self.save_sub_dir = save_dir.joinpath(save_sub_dir_name)
        self.save_sub_dir.mkdir(exist_ok=True)
        
        self.features_file_path = self.make_tensor_filepath("features")
        self.labels_file_path = self.make_tensor_filepath("labels")
        self.bin_values_file_path = self.make_tensor_filepath("bin_values")

        self.feature_names = [
            "q_squared", 
            "costheta_mu", 
            "costheta_K", 
            "chi"
        ]
        self.label_name = "dc9"
        self.binned_label_name = "dc9_bin_index"

        if regenerate:
            self.generate()

    def load(self):
        """
        Overwrite this with a function that loads some files
        (at least set self.features and self.labels).
        """

    def unload(self):
        """
        Overwrite this with a function that unloads the
        loaded data from memory (self.features and self.labels).
        """

    def generate(self):
        """
        Overwrite this with a function that saves some files
        (at least features and labels).
        """
        pass
    
    def make_tensor_filepath(self, kind):
        """
        Make a filepath for a torch tensor file.

        Parameters
        ----------
        kind : str
            The kind of tensor being saved.
            e.g. "labels".
        """
        file_name = (
            f"{self.extra_description}_{self.split}_{kind}.pt" 
            if self.extra_description 
            else f"{self.split}_{kind}.pt"
        )
        file_path = self.save_sub_dir.joinpath(file_name)
        return file_path

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Get a (feature, label) pair.

        Parameters
        ----------
        index : int
            The index of the pair within the dataset.

        Returns
        -------
        x : torch.Tensor
            The features.
        y : torch.Tensor
            The label.
        """
        x = self.features[index]
        y = self.labels[index]
        return x, y


class Binned_Signal_Dataset(Custom_Dataset):
    """For event-by-event approach."""
    def __init__(
            self, 
            level, 
            split, 
            save_dir, 
            q_squared_veto=True,
            std_scale=True,
            balanced_classes=True,
            shuffle=True,
            extra_description=None,
            regenerate=False,
    ):
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.shuffle = shuffle
        
        super().__init__(
            "binned_signal", 
            level, 
            q_squared_veto,
            split, 
            save_dir, 
            extra_description=extra_description,
            regenerate=regenerate,
        )

    def generate(self):    

        df_agg = load_aggregated_raw_signal(
            self.level, 
            self.split, 
            self.save_dir
        )
        
        def convert_to_binned_df(df_agg):
            bins, bin_values = to_bins(df_agg[self.label_name])
            df_agg[self.binned_label_name] = bins
            df_agg = df_agg.drop(columns=self.label_name)
            return df_agg, bin_values
        df_agg, bin_values = convert_to_binned_df(df_agg)
        
        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            q2_cut_strength = (
                "tight" if self.q_squared_veto
                else "loose"
            )
            df_agg = apply_q_squared_veto(df_agg, q2_cut_strength)
            if self.std_scale:
                for column_name in self.feature_names:
                    df_agg[column_name] = ( 
                        (
                            df_agg[column_name] 
                            - get_dataset_prescale("mean", self.level, self.q_squared_veto, column_name)
                        ) 
                        / get_dataset_prescale("std", self.level, self.q_squared_veto, column_name)
                    )
            if self.balanced_classes:
                df_agg = balance_classes(df_agg, self.binned_label_name)
            if self.shuffle:
                df_agg = df_agg.sample(frac=1)
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        features = torch.from_numpy(df_agg[self.feature_names].to_numpy())
        labels = torch.from_numpy(df_agg[self.binned_label_name].to_numpy())
        bin_values = torch.from_numpy(bin_values)

        torch.save(features, self.features_file_path)
        torch.save(labels, self.labels_file_path)
        torch.save(bin_values, self.bin_values_file_path)
        
        print(f"Generated features of shape: {features.shape}.")
        print(f"Generated labels of shape: {labels.shape}.")
        print(f"Generated bin values of shape: {bin_values.shape}.")

    def load(self):

        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        self.bin_values = torch.load(self.bin_values_file_path, weights_only=True)
        print(f"Loaded features of shape: {self.features.shape}.")
        print(f"Loaded labels of shape: {self.labels.shape}.")
        print(f"Loaded bin values of shape: {self.bin_values.shape}.")

    def unload(self):
        del self.features
        del self.labels
        del self.bin_values
        print("Unloaded data.")


class Signal_Sets_Dataset(Custom_Dataset):
    """Torch dataset of bootstrapped sets of signal events."""

    def __init__(
            self, 
            level, 
            split, 
            save_dir, 
            num_events_per_set,
            num_sets_per_label,
            binned=False,
            q_squared_veto=True,
            std_scale=True,
            balanced_classes=True,
            labels_to_sample=None,
            extra_description=None,
            regenerate=False,
    ):
        
        name = (
            "signal_sets_binned" if binned
            else "signal_sets_unbinned"
        )
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.binned = binned
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.labels_to_sample = labels_to_sample

        super().__init__(
            name, 
            level, 
            q_squared_veto,
            split, 
            save_dir, 
            extra_description=extra_description,
            regenerate=regenerate,
        )

    def generate(self):
        """
        Generate and save dataset state.
        """
        label_column_name = (
            self.binned_label_name if self.binned
            else self.label_name
        )

        df_agg = load_aggregated_raw_signal(
            self.level, 
            self.split, 
            self.save_dir
        )

        def convert_to_binned_df(df_agg):
            bins, bin_values = to_bins(df_agg[self.label_name])
            df_agg[self.binned_label_name] = bins
            df_agg = df_agg.drop(columns=self.label_name)
            return df_agg, bin_values
        if self.binned:
            df_agg, bin_values = convert_to_binned_df(df_agg)
            bin_values = torch.from_numpy(bin_values)

        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            q2_cut_strength = (
                "tight" if self.q_squared_veto
                else "loose"
            )
            df_agg = apply_q_squared_veto(df_agg, q2_cut_strength)
            if self.std_scale:
                for column_name in self.feature_names:
                    df_agg[column_name] = ( 
                        (
                            df_agg[column_name] 
                            - get_dataset_prescale("mean", self.level, self.q_squared_veto, column_name)
                        ) 
                        / get_dataset_prescale("std", self.level, self.q_squared_veto, column_name)
                    )
            if self.balanced_classes:
                df_agg = balance_classes(
                    df_agg, 
                    label_column_name=label_column_name
                )
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        source_features = torch.from_numpy(
            df_agg[self.feature_names]
            .to_numpy()
        )
        source_labels = torch.from_numpy(
            df_agg[label_column_name]
            .to_numpy()
        )

        if self.binned and self.labels_to_sample:
            self.labels_to_sample = [
                torch.argwhere(bin_values == label)
                .item() 
                for label in self.labels_to_sample
            ]

        features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=self.num_events_per_set, 
            m=self.num_sets_per_label,
            reduce_labels=True,
            labels_to_sample=self.labels_to_sample,
        )

        torch.save(
            features, 
            self.features_file_path
        )
        print(
            f"Generated features of shape: {features.shape}."
        )
        torch.save(
            labels, 
            self.labels_file_path
        )
        print(
            f"Generated labels of shape: {labels.shape}."
        )
        if self.binned:
            torch.save(
                bin_values, 
                self.bin_values_file_path
            )
            print(
               f"Generated bin values of shape: {bin_values.shape}."
            )

    def load(self): 
        """
        Load saved dataset state. 
        """
        self.features = torch.load(
            self.features_file_path, 
            weights_only=True
        )
        print(
            f"Loaded features of shape: {self.features.shape}."
        )
        self.labels = torch.load(
            self.labels_file_path, 
            weights_only=True
        )
        print(
            f"Loaded labels of shape: {self.labels.shape}."
        )
        if self.binned:
            self.bin_values = torch.load(
                self.bin_values_file_path,
                weights_only=True
            )
            print(
                f"Loaded bin values of shape: {self.bin_values.shape}."
            )

    def unload(self):
        del self.features
        del self.labels
        if self.binned:
            del self.bin_values
        print("Unloaded data.")


class Signal_Images_Dataset(Custom_Dataset):
    """Bootstrapped images (like Shawn)."""
    def __init__(
            self, 
            level, 
            split, 
            save_dir, 
            num_events_per_set,
            num_sets_per_label,
            n_bins,
            q_squared_veto=True,
            std_scale=True,
            balanced_classes=True,
            labels_to_sample=None,
            extra_description=None,
            regenerate=False,
    ):
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.n_bins = n_bins
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.labels_to_sample = labels_to_sample
        
        super().__init__(
            "signal_images", 
            level, 
            q_squared_veto,
            split, 
            save_dir, 
            extra_description=extra_description,
            regenerate=regenerate,
        )

    def generate(self):

        df_agg = load_aggregated_raw_signal(self.level, self.split, self.save_dir)
        
        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            q2_cut_strength = (
                "tight" if self.q_squared_veto
                else "loose"
            )
            df_agg = apply_q_squared_veto(df_agg, q2_cut_strength)
            if self.std_scale:
                df_agg["q_squared"] = (
                    (
                        df_agg["q_squared"] 
                        - get_dataset_prescale("mean", self.level, self.q_squared_veto, "q_squared")
                    )
                    / get_dataset_prescale("std", self.level, self.q_squared_veto, "q_squared")
                )
            if self.balanced_classes:
                df_agg = balance_classes(df_agg, label_column_name=self.label_name)
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        source_features = torch.from_numpy(df_agg[self.feature_names].to_numpy())
        source_labels = torch.from_numpy(df_agg[self.label_name].to_numpy())
        set_features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=self.num_events_per_set, m=self.num_sets_per_label,
            reduce_labels=True,
            labels_to_sample=self.labels_to_sample
        )
        features = torch.cat(
            [
                make_image(set_feat, n_bins=self.n_bins).unsqueeze(0) 
                for set_feat in set_features.numpy()
            ]
        )
        
        torch.save(features, self.features_file_path)
        torch.save(labels, self.labels_file_path)
        print(f"Generated features of shape: {features.shape}.")
        print(f"Generated labels of shape: {labels.shape}.")

    def load(self): 
        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        print(f"Loaded features of shape: {self.features.shape}.")
        print(f"Loaded labels of shape: {self.labels.shape}.")

    def unload(self):
        del self.features
        del self.labels
        print("Unloaded data.")


"""
Neural network training utilities.
"""


def print_gpu_memory_summary():
    print(torch.cuda.memory_summary(abbreviated=True))


def print_gpu_peak_memory_usage():

    def gpu_peak_memory_usage():
        return f"{torch.cuda.max_memory_allocated()/1024**3:.5f} GB"
    
    print(f"peak gpu memory usage: {gpu_peak_memory_usage()}")


def select_device():
    """
    Select a device to compute with.

    Returns
    -------
    str
        The name of the selected device.
        "cuda" if cuda is available,
        otherwise "cpu".
    """

    device = (
        "cuda" 
        if torch.cuda.is_available()
        else 
        "cpu"
    )
    print("Device: ", device)
    return device


def _train_batch(x, y, model, loss_fn, optimizer):
    """
    Train a model on a single batch given by x, y.
    
    Returns
    -------
    loss : float
    """
    model.train()
    
    yhat = model(x)    
    train_loss = loss_fn(yhat, y)

    train_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return train_loss


def _evaluate_batch(x, y, model, loss_fn):
    """
    Evaluate model on a mini-batch of data.
    """
    model.eval()
    with torch.no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
        return eval_loss
    

def _train_epoch(dataloader, model, loss_fn, optimizer, data_destination=None):
    """
    Train model over the dataset.
    """
    num_batches = len(dataloader)
    total_batch_loss = 0
    for x, y in dataloader:
        if data_destination is not None:
            x = x.to(data_destination)
            y = y.to(data_destination)
        batch_loss = _train_batch(x, y, model, loss_fn, optimizer)
        total_batch_loss += batch_loss
    avg_batch_loss = total_batch_loss / num_batches
    return avg_batch_loss
    

def _evaluate_epoch(dataloader, model, loss_fn, data_destination=None, scheduler=None):
    """
    Evaluate model over the dataset.
    """
    num_batches = len(dataloader)
    total_batch_loss = 0
    for x, y in dataloader:
        if data_destination is not None:
            x = x.to(data_destination)
            y = y.to(data_destination)
        batch_loss = _evaluate_batch(x, y, model, loss_fn)
        total_batch_loss += batch_loss
    avg_batch_loss = total_batch_loss / num_batches
    if scheduler:
        scheduler.step(avg_batch_loss)
    return avg_batch_loss


def _print_epoch_loss(epoch, train_loss, eval_loss):
    """
    Print a summary of loss values for an epoch.
    """
    print(f"\nepoch {epoch} complete:")
    print(f"    Train loss: {train_loss}")
    print(f"    Eval loss: {eval_loss}\n")


def _print_scheduler_last_learning_rate(scheduler):
    last_learning_rate = scheduler.get_last_lr()
    message = f"learning rate: {last_learning_rate}"
    print(message)


def train_and_eval(
    model, 
    train_dataset, eval_dataset,
    loss_fn, optimizer, 
    epochs, train_batch_size, eval_batch_size, 
    device, move_data=True,
    scheduler=None,
    checkpoint_epochs=10
):
    """
    Train and evaluate a model.
    """

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        drop_last=True, 
        shuffle=True
    ) #, pin_memory=True, num_workers=4)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=eval_batch_size, 
        drop_last=True, 
        shuffle=True
    ) # , pin_memory=True, num_workers=4)
    
    model = model.to(device)
    data_destination = (device if move_data else None)

    for ep in range(epochs):
        train_loss = _train_epoch(train_dataloader, model, loss_fn, optimizer, data_destination=data_destination).item()
        eval_loss = _evaluate_epoch(eval_dataloader, model, loss_fn, data_destination=data_destination, scheduler=scheduler).item()
        model.append_to_loss_table(ep, train_loss, eval_loss)
        _print_epoch_loss(ep, train_loss, eval_loss)
        if scheduler:
            _print_scheduler_last_learning_rate(scheduler)
        print_gpu_peak_memory_usage()
        if (ep % checkpoint_epochs == 0):
            model.save_checkpoint(ep)

    model.save_final()    
    model.save_loss_table()


def plot_loss_curves(loss_table, ax, start_epoch=0, log_scale=False):
    """
    Plot loss curves given a loss table.

    Parameters
    ----------
    loss_table : dict
        Dictionary with keys "epoch", "train_loss", and "eval_loss".
    start_epoch : int
        First epoch to plot. (Previous epochs are not plotted.)
    ax : matplotlib axes
        Axes on which to plot.
    """

    epochs_to_plot = loss_table["epoch"][start_epoch:]
    train_losses_to_plot = loss_table["train_loss"][start_epoch:]
    eval_losses_to_plot = loss_table["eval_loss"][start_epoch:]
    
    ax.plot(
        epochs_to_plot, 
        train_losses_to_plot, 
        label="Training Loss"
    )
    ax.plot(
        epochs_to_plot, 
        eval_losses_to_plot, 
        label="Eval. Loss"
    )
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Epoch")


"""
Neural network models.
"""


class Custom_Model(torch.nn.Module):
    """Custom model."""
    def __init__(self, kind, save_dir, extra_description=None):
        """
        save_dir : str
            Directory where all models are saved.
            Model will be saved in a subdirectory of
            the save_dir directory.
        """
        super().__init__()

        self.kind = kind
        self.extra_description = extra_description

        self.save_sub_dir = pathlib.Path(save_dir).joinpath(
            f"{self.kind}_{self.extra_description}" if extra_description
            else self.kind
        )
        
        self.loss_table = self.make_empty_loss_table()
        
    def make_final_save_path(self):
        file_name = "final.pt"
        file_path = self.save_sub_dir.joinpath(file_name)
        return file_path 
    
    def save_final(self):
        file_path = self.make_final_save_path()
        torch.save(self.state_dict(), file_path)

    def load_final(self):
        model_file_path = self.make_final_save_path()
        self.load_state_dict(torch.load(model_file_path, weights_only=True))
        self.loss_table = self.load_loss_table()
    
    def make_checkpoint_save_path(self, epoch:int):
        file_name = f"epoch_{epoch}.pt"
        file_path = self.save_sub_dir.joinpath(file_name)
        return file_path
        
    def save_checkpoint(self, epoch):
        file_path = self.make_checkpoint_save_path(epoch)
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, epoch):
        file_path = self.make_checkpoint_save_path(epoch)
        self.load_state_dict(torch.load(file_path, weights_only=True))

    def make_loss_table_file_path(self):
        file_name = "loss_table.pkl"
        file_path = self.save_sub_dir.joinpath(file_name)
        return file_path
    
    def save_loss_table(self):
        file_path = self.make_loss_table_file_path()
        with open(file_path, "wb") as handle:
            pickle.dump(self.loss_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_loss_table(self):
        file_path = self.make_loss_table_file_path()
        with open(file_path, "rb") as handle:
            loss_table = pickle.load(handle)
        return loss_table
    
    def append_to_loss_table(self, epoch, train_loss, eval_loss):
        self.loss_table["epoch"].append(epoch)
        self.loss_table["train_loss"].append(train_loss)
        self.loss_table["eval_loss"].append(eval_loss)
        assert (
            len(self.loss_table["epoch"]) 
            == len(self.loss_table["train_loss"]) 
            == len(self.loss_table["eval_loss"])
        )

    def make_empty_loss_table(self):
        """Create an empty loss table."""
        empty_loss_table = {"epoch":[], "train_loss":[], "eval_loss":[]}
        return empty_loss_table
    
    def clear_loss_table(self):
        self.loss_table = self.make_empty_loss_table()

    def retrain(
        self,
        train_dataset,
        eval_dataset,
        loss_fn,
        optimizer,
        epochs,
        train_batch_size,
        eval_batch_size,
        device,
        move_data=True,
        scheduler=None,
        checkpoint_epochs=5,
    ):
        
        try: self.save_sub_dir.mkdir()
        except FileExistsError:
            error_message = (
                "Model already exists."
                + " Delete existing model (subdirectory) to continue."
            )
            raise FileExistsError(error_message)
        
        train_and_eval(
            model=self, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
            move_data=move_data,
            scheduler=scheduler,
            checkpoint_epochs=checkpoint_epochs,
        )
        

class CNN_Res(Custom_Model):
    
    def __init__(self, save_dir, extra_description=None):
        super().__init__("cnn_res", save_dir, extra_description=extra_description)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=1, 
                padding="same", 
                bias=False
            ),
            # torch.nn.BatchNorm3d(num_features=16),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=1, padding=1),
            *[self.Res_Block(in_out_channels=16) for _ in range(3)],
            self.Conv_Block(in_channels=16, out_channels=16),
            *[self.Res_Block(in_out_channels=16) for _ in range(3)],
            self.Conv_Block(in_channels=16, out_channels=16),
            *[self.Res_Block(in_out_channels=16) for _ in range(3)],
            # Conv_Block(in_channels=128, out_channels=128),
            # *[Res_Block(in_out_channels=128) for _ in range(1)],
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(in_features=16, out_features=32),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=32, out_features=1),
        )
        
        self.double()

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=(2,3,4))
        x = self.dense(x)
        x = torch.squeeze(x)
        return x

    class Res_Block(torch.nn.Module):
        def __init__(self, in_out_channels):
            super().__init__()
            
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_out_channels, 
                    out_channels=in_out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # torch.nn.BatchNorm3d(num_features=in_out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    in_channels=in_out_channels, 
                    out_channels=in_out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # torch.nn.BatchNorm3d(num_features=in_out_channels),
            )
            self.last_activation = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.block(x) + x
            x = self.last_activation(x)
            return x

    class Conv_Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            
            super().__init__()
            
            self.block_a = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # torch.nn.BatchNorm3d(num_features=out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, stride=1, padding="same"
                ),
                # torch.nn.BatchNorm3d(num_features=out_channels),
            )
            
            self.block_b = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                # torch.nn.BatchNorm3d(num_features=out_channels),
            )
            
            self.last_activation = torch.nn.ReLU()
        
        def forward(self, x):
            out_block_a = self.block_a(x)
            out_block_b = self.block_b(x)
            x = out_block_a + out_block_b
            x = self.last_activation(x)
            return x


class Deep_Sets(Custom_Model):

    def __init__(self, save_dir, extra_description=None):
        super().__init__("deep_sets", save_dir, extra_description=extra_description)

        self.f = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=32),
            # torch.nn.LayerNorm(normalized_shape=(num_events_per_set, 32)),
            torch.nn.ReLU(),
            *[self.Res_Block_Event(in_out_features=32) for _ in range(3)],
            torch.nn.Linear(in_features=32, out_features=32),
            # torch.nn.LayerNorm(normalized_shape=(num_events_per_set,32)),
        )

        self.g = torch.nn.Sequential(
            *[self.Res_Block_Set(in_out_features=32) for _ in range(3)],
            torch.nn.Linear(in_features=32, out_features=32),
            # torch.nn.BatchNorm1d(num_features=32), 
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=1),
        )
        
        self.double()

    def forward(self, x):
        x = self.f(x)
        x = torch.mean(x, dim=1)
        x = self.g(x)
        x = torch.squeeze(x)
        return x
    
    class Res_Block_Event(torch.nn.Module):
        def __init__(self, in_out_features):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # torch.nn.LayerNorm(normalized_shape=(num_events_per_set, in_out_features)),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # torch.nn.LayerNorm(normalized_shape=(num_events_per_set, in_out_features)),
            )
            self.last_activation = torch.nn.ReLU()
        def forward(self, x):
            x = self.block(x) #+ x
            x = self.last_activation(x)
            return x

    class Res_Block_Set(torch.nn.Module):
        def __init__(self, in_out_features):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # torch.nn.BatchNorm1d(num_features=in_out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=in_out_features, out_features=in_out_features),
                # torch.nn.BatchNorm1d(num_features=in_out_features),
            )
            self.last_activation = torch.nn.ReLU()
        def forward(self, x):
            x = self.block(x) #+ x
            x = self.last_activation(x)
            return x
    

class Event_By_Event_NN(Custom_Model):
    def __init__(self, save_dir, extra_description=None):
        super().__init__("event_by_event_nn", save_dir, extra_description=extra_description)

        self.base = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 44),
        )
        
        self.double()

    def forward(self, x):
        event_logits = self.base(x)
        return event_logits
    
    def predict_log_probabilities(self, x):
        """
        Predict the log probability of each class, given a set of events.

        x : A torch tensor of features of events. (a set)
        """
        with torch.no_grad():
            event_logits = self.forward(x)
            event_log_probabilities = torch.nn.functional.log_softmax(event_logits, dim=1)
            set_logits = torch.sum(event_log_probabilities, dim=0)
            set_log_probabilities = torch.nn.functional.log_softmax(set_logits, dim=0)
        return set_log_probabilities
    
    def calculate_expected_value(self, x, bin_values):
        """
        Calculate the prediction expectation value, given a set of events.

        x : A torch tensor of features of events. (a set)
        """
        with torch.no_grad():
            bin_shift = 5
            bin_values = bin_values + bin_shift
            log_bin_values = torch.log(bin_values)
            log_probs = self.predict_log_probabilities(x)
            lse = torch.logsumexp(log_bin_values + log_probs, dim=0)
            yhat = torch.exp(lse) - bin_shift
        return yhat
    


"""
Model evaluation utilities.
"""

class Summary_Table:
    def __init__(self):
        self.table = self.make_empty()

    def add_item(
        self, 
        level,
        q_squared_veto:bool,
        method_name, 
        item_name, 
        num_events_per_set, 
        item,
    ):
        if type(item) is torch.Tensor:
            item = item.item()
        self.table.loc[
            (level, q_squared_veto, method_name, num_events_per_set), 
            item_name,
        ] = item
    
    def reset_table(self):
        self.table = self.make_empty()
    
    def make_empty(self):
        index = pandas.MultiIndex.from_product(
            [
                ["gen", "det"],
                [True, False],
                [
                    "Images", 
                    "Deep Sets", 
                    "Event by event"    
                ],
                [70_000, 24_000, 6_000],
            ],
            names=["Level", "q2_veto", "Method", "Events/set"]
        )
        table = pandas.DataFrame(
            index=index, 
            columns=[
                "MSE",
                "MAE", 
                "Std. at NP", 
                "Mean at NP", 
                "Bias at NP"
            ]
        )
        return table


def make_predictions(
    model, 
    features, 
    device, 
    event_by_event=False, 
    bin_values=None
):
    """
    Make predictions on an array of features.

    Features should be an array of sets of events.
    Bin values must be specified for event-by-event method.
    """
    with torch.no_grad():
        predictions = []
        for feat in features:
            if event_by_event:
                assert bin_values is not None
                prediction = model.calculate_expected_value(
                    feat.to(device),
                    bin_values.to(device),
                )
            else:
                prediction = model(feat.unsqueeze(0).to(device))
            predictions.append(prediction)
        predictions = torch.tensor(predictions)
    return predictions


def run_linearity_test(predictions, labels):
    """
    Calculate the average and standard deviation of
    predictions for each label.

    DANGER: Assumes data sorted by labels.
    
    Returns
    -------
    unique_labels : ...
    avg_yhat_per_label : ...
    std_yhat_per_label : ...
    """

    with torch.no_grad():
        num_sets_per_label = get_num_per_unique_label(labels)
        avg_yhat_per_label = predictions.reshape(-1, num_sets_per_label).mean(dim=1)
        std_yhat_per_label = predictions.reshape(-1, num_sets_per_label).std(dim=1)
        unique_labels = torch.unique(labels)
    return unique_labels, avg_yhat_per_label, std_yhat_per_label


def run_sensitivity_test(predictions, label):
    """
    Find the standard deviation and mean of predictions for
    a single label.

    Returns
    -------
    mean : ...
    std : ...
    bias : ...
    """
    mean = predictions.mean()
    std = predictions.std()
    bias = mean - label
    return mean, std, bias


def calculate_mse_mae(predictions, labels):
    """
    Calculate the mean squared error and the mean absolute error.

    Returns
    -------
    mse : ...
    mae : ...
    """
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(predictions, labels)
        mae = torch.nn.functional.l1_loss(predictions, labels)
    return mse, mae


def plot_prediction_linearity(
    ax,
    input_values, 
    avg_pred, 
    stdev_pred, 
    ref_line_buffer=0.05, 
    xlim=(-2.25, 1.35), 
    ylim=(-2.25, 1.35), 
    xlabel=r"Actual $\delta C_9$", 
    ylabel=r"Predicted $\delta C_9$",
    note=None,
):
    """
    input_values : array 
        value corresponding to each bin index
    avg_pred : array
        ndarray of average prediction per input bin
    stdev_pred : array
        ndarray of standard deviation of prediction per input bin 
    ref_line_buffer : float 
        extra amount to extend reference line
    xlim : tuple
        x limits
    ylim : tuple
        y limits
    xlabel : str
        x axis label
    ylabel : str
        y axis label
    note : str
        note to add
    """
        
    ax.scatter(
        input_values, 
        avg_pred, 
        label="Avg.", 
        color="firebrick", 
        s=16, 
        zorder=5
    )
    ax.errorbar(
        input_values, 
        avg_pred, 
        yerr=stdev_pred, 
        fmt="none", 
        elinewidth=0.5, 
        capsize=0.5, 
        color="black", 
        label="Std. Dev.", 
        zorder=10
    )

    ref_ticks = numpy.linspace(
        numpy.min(input_values)-ref_line_buffer, 
        numpy.max(input_values)+ref_line_buffer, 
        2,
    )
    ax.plot(
        ref_ticks, 
        ref_ticks,
        label="Ref. Line (Slope = 1)",
        color="grey",
        linewidth=0.5,
        zorder=0,
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    make_plot_note(ax, note)


def plot_sensitivity(
    ax, 
    predictions,
    label,
    bins=50, 
    xbounds=(-1.5, 0), 
    ybounds=(0, 200), 
    std_marker_height=20,
    note=None,
):
    
    (
        mean, 
        std, 
        bias
    ) = run_sensitivity_test(
        predictions, 
        label
    )

    ax.hist(
        predictions, 
        bins=bins, 
        range=xbounds
    )
    ax.vlines(
        label,
        0,
        ybounds[1],
        color="red",
        label=f"Target ({label})",
    )
    ax.vlines(
        mean,
        0,
        ybounds[1],
        color="red",
        linestyles="--",
        label=r"$\mu = $ " + f"{mean.round(decimals=3)}"
    )
    ax.hlines(
        std_marker_height,
        mean,
        mean+std,
        color="orange",
        linestyles="dashdot",
        label=r"$\sigma = $ " + f"{std.round(decimals=3)}"
    )
    ax.set_xlabel(r"Predicted $\delta C_9$")
    ax.set_xbound(*xbounds)
    ax.set_ybound(*ybounds)
    ax.legend()
    make_plot_note(
        ax, 
        note, 
        fontsize="medium"
    )


"""
End-to-end high level functionality for evaluating an approach.
This includes dataset generation, training, and evaluation.
"""

class Shawns_Approach:
    def __init__(
        self,
        device,
        level,
        datasets_dir,
        models_dir,
        plots_dir,
        summary_table,
        set_sizes=[70_000, 24_000, 6_000],
        new_physics_delta_c9_value=-0.82,
        num_image_bins=10,
        num_sets_per_label=50,
        num_sets_per_label_sensitivity=2000,
        regenerate_datasets=False,
        balanced_classes=True,
        q_squared_veto=True,
        std_scale=True,
        retrain_models=False,
        learning_rate=4e-4,
        epochs=80,
        train_batch_size=32,
        eval_batch_size=32,
    ):
        
        self.device = device
        self.level = level
        self.models_dir = pathlib.Path(models_dir)
        self.dataset_dir = pathlib.Path(datasets_dir)
        self.plots_dir = pathlib.Path(plots_dir)
        self.summary_table = summary_table
        self.set_sizes = set_sizes
        self.new_physics_delta_c9_value = new_physics_delta_c9_value
        self.num_image_bins = num_image_bins
        self.num_sets_per_label = num_sets_per_label
        self.num_sets_per_label_sensitivity = num_sets_per_label_sensitivity
        self.regenerate_datasets = regenerate_datasets
        self.balanced_classes = balanced_classes
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.retrain_models = retrain_models
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.init_datasets()
        self.peek_at_features()
        self.init_models()
        self.evaluate_models()

    def init_datasets(self):

        self.train_datasets = {
            num_events_per_set : Signal_Images_Dataset(
                level=self.level, 
                split="train", 
                save_dir=self.dataset_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_sets_per_label,
                n_bins=self.num_image_bins,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                extra_description=f"{num_events_per_set}",
                regenerate=self.regenerate_datasets,
            ) 
            for num_events_per_set in self.set_sizes
        }

        self.eval_datasets = {
            num_events_per_set : Signal_Images_Dataset(
                level=self.level, 
                split="eval", 
                save_dir=self.dataset_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_sets_per_label,
                n_bins=self.num_image_bins,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                extra_description=f"{num_events_per_set}",
                regenerate=self.regenerate_datasets,
            ) 
            for num_events_per_set in self.set_sizes
        }

        self.single_label_eval_datasets = {
            num_events_per_set : Signal_Images_Dataset(
                level=self.level, 
                split="eval", 
                save_dir=self.dataset_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_sets_per_label_sensitivity,
                n_bins=self.num_image_bins,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                labels_to_sample=[self.new_physics_delta_c9_value],
                extra_description=f"{num_events_per_set}_single",
                regenerate=self.regenerate_datasets,
            ) 
            for num_events_per_set in self.set_sizes
        }

    def peek_at_features(self):

        num_events_per_set = self.set_sizes[1]
        dset = self.train_datasets[num_events_per_set]
        dset.load()

        plot_image_slices(
            dset.features[0], 
            n_slices=3, 
            note=r"$\delta C_9$ : "+f"{dset.labels[0]}"
        )
        plt.show()
        plt.close()

        dset.unload()

    def init_models(self):

        self.models = {
            num_events_per_set : CNN_Res(
                self.models_dir, 
                extra_description=f"{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}"
            )
            for num_events_per_set in self.set_sizes
        }

        if self.retrain_models:
            self.train_models()

    def train_models(self):

        for num_events_per_set in self.set_sizes:

            self.train_model(num_events_per_set)

    def train_model(self, num_events_per_set):

        model = self.models[
            num_events_per_set
        ]

        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate
        )

        train_dataset = self.train_datasets[
            num_events_per_set
        ]
        eval_dataset = self.eval_datasets[
            num_events_per_set
        ]
        train_dataset.load()
        eval_dataset.load()

        model.retrain( 
            train_dataset, 
            eval_dataset, 
            loss_fn, 
            optimizer, 
            self.epochs, 
            self.train_batch_size, 
            self.eval_batch_size, 
            self.device, 
            move_data=True,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                factor=0.9, 
                patience=1
            ),
            checkpoint_epochs=5,
        )

        _, ax = plt.subplots()
        plot_loss_curves(
            model.loss_table,
            ax,
            start_epoch=0,
            log_scale=True,
        )
        plt.show()
        plt.close()

        plot_file_name = f"images_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_loss.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        train_dataset.unload()
        eval_dataset.unload()

    def evaluate_models(self,):

        for num_events_per_set in self.set_sizes:

            self.evaluate_model(num_events_per_set)

    def evaluate_model(self, num_events_per_set):
        
        model = self.models[
            num_events_per_set
        ]
        model.load_final()
        model.to(self.device)
        model.eval()

        self.evaluate_mse_mae(model, num_events_per_set)
        self.evaluate_linearity(model, num_events_per_set)
        self.evaluate_sensitivity(model, num_events_per_set)

    def evaluate_mse_mae(self, model, num_events_per_set):

        eval_dataset = self.eval_datasets[
            num_events_per_set
        ]
        eval_dataset.load()

        predictions = make_predictions(
            model, 
            eval_dataset.features,
            self.device,
        )

        mse, mae = calculate_mse_mae(
            predictions, 
            eval_dataset.labels,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Images", 
            "MSE", 
            num_events_per_set, 
            mse,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Images", 
            "MAE", 
            num_events_per_set, 
            mae,
        )
    
    def evaluate_linearity(self, model, num_events_per_set,):

        eval_dataset = self.eval_datasets[
            num_events_per_set
        ]
        eval_dataset.load()

        predictions = make_predictions(
            model, 
            eval_dataset.features,
            self.device,
        )

        (
            unique_labels, 
            avgs, 
            stds,
        ) = run_linearity_test(
            predictions, 
            eval_dataset.labels
        )
        _, ax = plt.subplots()
        plot_prediction_linearity(
            ax,
            unique_labels.detach().cpu().numpy(),
            avgs.detach().cpu().numpy(),
            stds.detach().cpu().numpy(),
            note=(
                f"Images ({self.num_image_bins} bins), {self.level}., "
                + f"{self.num_sets_per_label} boots., "
                + f"{num_events_per_set} events/boots. "
                + f"$q^2$ veto: {self.q_squared_veto}"
            ),
        )

        plot_file_name = f"images_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_lin.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()

        eval_dataset.unload()

    def evaluate_sensitivity(self, model, num_events_per_set,):

        single_label_eval_dataset = self.single_label_eval_datasets[
            num_events_per_set
        ]
        single_label_eval_dataset.load()

        single_label_predictions = make_predictions(
            model, 
            single_label_eval_dataset.features,
            self.device,
        )

        mean, std, bias = run_sensitivity_test(
            single_label_predictions, 
            self.new_physics_delta_c9_value,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Images", 
            "Mean at NP", 
            num_events_per_set, 
            mean,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Images", 
            "Std. at NP", 
            num_events_per_set, 
            std
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Images", 
            "Bias at NP", 
            num_events_per_set, 
            bias
        )

        single_label_eval_dataset.unload()

        _, ax = plt.subplots()

        plot_sensitivity(
            ax,
            single_label_predictions,
            self.new_physics_delta_c9_value,
            note=(
                f"Images ({self.num_image_bins} bins), {self.level}., " 
                + f"{self.num_sets_per_label_sensitivity} boots., " 
                + f"{num_events_per_set} events/boots. "
                + f"$q^2$ veto: {self.q_squared_veto}"
            ), 
        )

        plot_file_name = f"images_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_sens.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()


class Deep_Sets_Approach:
    def __init__(
        self,
        device,
        level,
        datasets_dir,
        models_dir,
        plots_dir,
        summary_table,
        set_sizes=[70_000, 24_000, 6_000],
        new_physics_delta_c9_value=-0.82,
        num_sets_per_label=50,
        num_sets_per_label_sensitivity=2000,
        regenerate_datasets=False,
        balanced_classes=True,
        q_squared_veto=True,
        std_scale=True,
        retrain_models=False,
        learning_rate=4e-4,
        epochs=80,
        train_batch_size=32,
        eval_batch_size=32,
    ):
        self.device = device
        self.level = level
        self.models_dir = pathlib.Path(models_dir)
        self.dataset_dir = pathlib.Path(datasets_dir)
        self.plots_dir = pathlib.Path(plots_dir)
        self.summary_table = summary_table
        self.set_sizes = set_sizes
        self.new_physics_delta_c9_value = new_physics_delta_c9_value
        self.num_sets_per_label = num_sets_per_label
        self.num_sets_per_label_sensitivity = num_sets_per_label_sensitivity
        self.regenerate_datasets = regenerate_datasets
        self.balanced_classes = balanced_classes
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.retrain_models = retrain_models
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.init_datasets()
        self.init_models()
        self.evaluate_models()

    def init_datasets(self):

        self.train_datasets = {
            num_events_per_set : Signal_Sets_Dataset(
                level=self.level,
                split="train",
                save_dir=self.dataset_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_sets_per_label,
                binned=False,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                extra_description=f"{num_events_per_set}",
                regenerate=self.regenerate_datasets
            )
            for num_events_per_set in self.set_sizes
        }

        self.eval_datasets = {
            num_events_per_set : Signal_Sets_Dataset(
                level=self.level,
                split="eval",
                save_dir=self.dataset_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_sets_per_label,
                binned=False,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                extra_description=f"{num_events_per_set}",
                regenerate=self.regenerate_datasets
            )
            for num_events_per_set in self.set_sizes
        }

        self.single_label_eval_datasets = {
            num_events_per_set : Signal_Sets_Dataset(
                level=self.level,
                split="eval",
                save_dir=self.dataset_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_sets_per_label_sensitivity,
                binned=False,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                labels_to_sample=[self.new_physics_delta_c9_value],
                extra_description=f"{num_events_per_set}_single",
                regenerate=self.regenerate_datasets
            )
            for num_events_per_set in self.set_sizes
        }

    def init_models(self):

        self.models = {
            num_events_per_set : Deep_Sets(
                self.models_dir, 
                extra_description=f"{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}"
            )
            for num_events_per_set in self.set_sizes
        }

        if self.retrain_models:
            self.train_models()

    def train_models(self):

        for num_events_per_set in self.set_sizes:

            self.train_model(num_events_per_set)

    def train_model(self, num_events_per_set):

        model = self.models[
            num_events_per_set
        ]

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate
        )

        train_dataset = self.train_datasets[
            num_events_per_set
        ]
        eval_dataset = self.eval_datasets[
            num_events_per_set
        ]
        train_dataset.load()
        eval_dataset.load()

        model.retrain( 
            train_dataset, 
            eval_dataset, 
            loss_fn, 
            optimizer, 
            self.epochs, 
            self.train_batch_size, 
            self.eval_batch_size, 
            self.device, 
            move_data=True,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                factor=0.9, 
                patience=1
            ),
            checkpoint_epochs=5,
        )

        _, ax = plt.subplots()
        plot_loss_curves(
            model.loss_table,
            ax,
            start_epoch=0,
            log_scale=True,
        )

        plot_file_name = f"deepsets_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_loss.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()

        train_dataset.unload()
        eval_dataset.unload()
    
    def evaluate_models(self,):

        for num_events_per_set in self.set_sizes:

            self.evaluate_model(num_events_per_set)

    def evaluate_model(self, num_events_per_set):
        
        model = self.models[
            num_events_per_set
        ]
        model.load_final()
        model.to(self.device)
        model.eval()

        self.evaluate_mse_mae(model, num_events_per_set)
        self.evaluate_linearity(model, num_events_per_set)
        self.evaluate_sensitivity(model, num_events_per_set)

    def evaluate_mse_mae(self, model, num_events_per_set):

        eval_dataset = self.eval_datasets[
            num_events_per_set
        ]
        eval_dataset.load()

        predictions = make_predictions(
            model, 
            eval_dataset.features,
            self.device,
        )

        mse, mae = calculate_mse_mae(
            predictions, 
            eval_dataset.labels,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Deep Sets", 
            "MSE", 
            num_events_per_set, 
            mse,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Deep Sets", 
            "MAE", 
            num_events_per_set, 
            mae,
        )
    
    def evaluate_linearity(self, model, num_events_per_set,):

        eval_dataset = self.eval_datasets[
            num_events_per_set
        ]
        eval_dataset.load()

        predictions = make_predictions(
            model, 
            eval_dataset.features,
            self.device,
        )

        (
            unique_labels, 
            avgs, 
            stds,
        ) = run_linearity_test(
            predictions, 
            eval_dataset.labels
        )

        _, ax = plt.subplots()
        plot_prediction_linearity(
            ax,
            unique_labels.detach().cpu().numpy(),
            avgs.detach().cpu().numpy(),
            stds.detach().cpu().numpy(),
            note=(
                f"Deep Sets, {self.level}., "
                + f"{self.num_sets_per_label} boots., "
                + f"{num_events_per_set} events/boots."
                + f"$q^2$ veto: {self.q_squared_veto}"
            ),
        )

        plot_file_name = f"deepsets_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_lin.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()

        eval_dataset.unload()

    def evaluate_sensitivity(self, model, num_events_per_set,):

        single_label_eval_dataset = self.single_label_eval_datasets[
            num_events_per_set
        ]
        single_label_eval_dataset.load()

        single_label_predictions = make_predictions(
            model, 
            single_label_eval_dataset.features,
            self.device,
        )

        mean, std, bias = run_sensitivity_test(
            single_label_predictions, 
            self.new_physics_delta_c9_value,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Deep Sets", 
            "Mean at NP", 
            num_events_per_set, 
            mean,
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Deep Sets", 
            "Std. at NP", 
            num_events_per_set, 
            std
        )
        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Deep Sets", 
            "Bias at NP", 
            num_events_per_set, 
            bias
        )

        single_label_eval_dataset.unload()

        _, ax = plt.subplots()

        plot_sensitivity(
            ax,
            single_label_predictions,
            self.new_physics_delta_c9_value,
            note=(
                f"Deep Sets, {self.level}., " 
                + f"{self.num_sets_per_label_sensitivity} boots., " 
                + f"{num_events_per_set} events/boots."
                + f"$q^2$ veto: {self.q_squared_veto}"
            ), 
        )

        plot_file_name = f"deepsets_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_sens.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()


class Event_By_Event_Approach:
    def __init__(
        self,
        device,
        level,
        datasets_dir,
        models_dir,
        plots_dir,
        summary_table,
        set_sizes=[70_000, 24_000, 6_000],
        new_physics_delta_c9_value=-0.82,
        num_eval_sets_per_label=50,
        num_eval_sets_per_label_sensitivity=2000,
        regenerate_datasets=False,
        balanced_classes=True,
        q_squared_veto=True,
        std_scale=True,
        retrain_model=False,
        learning_rate=3e-3,
        epochs=200,
        train_batch_size=10_000,
        eval_batch_size=10_000,
    ):
        
        self.device = device
        self.level = level
        self.datasets_dir = pathlib.Path(datasets_dir)
        self.models_dir = pathlib.Path(models_dir)
        self.plots_dir = pathlib.Path(plots_dir)
        self.summary_table = summary_table
        self.set_sizes = set_sizes
        self.new_physics_delta_c9_value = new_physics_delta_c9_value
        self.num_eval_sets_per_label = num_eval_sets_per_label
        self.num_eval_sets_per_label_sensitivity = num_eval_sets_per_label_sensitivity
        self.regenerate_datasets = regenerate_datasets
        self.balanced_classes = balanced_classes
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.retrain_model = retrain_model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_batch_size= train_batch_size
        self.eval_batch_size = eval_batch_size

        self.init_datasets()
        self.init_model()
        self.evaluate_model()


    def init_datasets(self):

        self.train_events_dataset = Binned_Signal_Dataset(
            level=self.level,
            split="train",
            save_dir=self.datasets_dir,
            q_squared_veto=self.q_squared_veto,
            std_scale=self.std_scale,
            balanced_classes=self.balanced_classes,
            shuffle=True,
            extra_description=None,
            regenerate=self.regenerate_datasets
        )

        self.eval_events_dataset = Binned_Signal_Dataset(
            level=self.level,
            split="eval",
            save_dir=self.datasets_dir,
            q_squared_veto=self.q_squared_veto,
            std_scale=self.std_scale,
            balanced_classes=self.balanced_classes,
            shuffle=True,
            extra_description=None,
            regenerate=self.regenerate_datasets
        )

        self.eval_sets_datasets = {
            num_events_per_set : Signal_Sets_Dataset(
                level=self.level,
                split="eval",
                save_dir=self.datasets_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_eval_sets_per_label,
                binned=True,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                extra_description=f"{num_events_per_set}",
                regenerate=self.regenerate_datasets
            )
            for num_events_per_set in self.set_sizes
        }

        self.single_label_eval_sets_datasets = {
            num_events_per_set : Signal_Sets_Dataset(
                level=self.level,
                split="eval",
                save_dir=self.datasets_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_eval_sets_per_label_sensitivity,
                binned=True,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                labels_to_sample=[self.new_physics_delta_c9_value],
                extra_description=f"{num_events_per_set}_single",
                regenerate=self.regenerate_datasets
            )
            for num_events_per_set in self.set_sizes
        }

    def init_model(self):
        self.model = Event_By_Event_NN(
            self.models_dir, 
            extra_description=f"{self.level}_q2v_{self.q_squared_veto}",
        )

        if self.retrain_model:
            self.train_model()

    def train_model(self):
        
        model = self.model

        train_dataset = self.train_events_dataset
        eval_dataset = self.eval_events_dataset
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.95, 
            threshold=0, 
            patience=0, 
            eps=1e-9
        )
        
        train_dataset.load()
        eval_dataset.load()
        
        model.retrain(
            train_dataset,
            eval_dataset,
            loss_fn,
            optimizer,
            self.epochs,
            self.train_batch_size,
            self.eval_batch_size,
            self.device,
            scheduler=scheduler,
        )
        
        _, ax = plt.subplots()
        plot_loss_curves(
            model.loss_table,
            ax,
            start_epoch=0,
            log_scale=True,
        )

        plot_file_name = f"ebe_{self.level}_q2v_{self.q_squared_veto}_loss.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()

        train_dataset.unload()
        eval_dataset.unload()

    
    def evaluate_model(self):

        model = self.model
        model.load_final()
        model.to(self.device)
        model.eval()

        for num_events_per_set in self.set_sizes:
            self.evaluate_mse_mae(num_events_per_set)
            self.evaluate_linearity(num_events_per_set)
            self.evaluate_sensitivity(num_events_per_set)

    def evaluate_mse_mae(self, num_events_per_set):
        
        eval_sets_dataset = self.eval_sets_datasets[
            num_events_per_set
        ]
        eval_sets_dataset.load()

        predictions = make_predictions(
            self.model, 
            eval_sets_dataset.features, 
            self.device, 
            event_by_event=True, 
            bin_values=eval_sets_dataset.bin_values
        )
        assert (
           predictions.shape 
           == eval_sets_dataset.labels.shape
        )

        unbinned_labels = (
            eval_sets_dataset
            .bin_values[
                eval_sets_dataset.labels.int()
            ]
        )

        (
            mse, 
            mae
        ) = calculate_mse_mae(
            predictions, 
            unbinned_labels
        )

        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Event by event", 
            "MSE", 
            num_events_per_set, 
            mse
        )

        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Event by event", 
            "MAE", 
            num_events_per_set, 
            mae
        )

    def evaluate_linearity(self, num_events_per_set):

        eval_sets_dataset = self.eval_sets_datasets[
            num_events_per_set
        ]
        eval_sets_dataset.load()

        predictions = make_predictions(
            self.model, 
            eval_sets_dataset.features, 
            self.device, 
            event_by_event=True, 
            bin_values=eval_sets_dataset.bin_values
        )
        assert (
           predictions.shape 
           == eval_sets_dataset.labels.shape
        )

        unbinned_labels = (
           eval_sets_dataset
            .bin_values[
                eval_sets_dataset.labels.int()
            ]
        )

        (
            unique_labels, 
            avgs, 
            stds
        ) = run_linearity_test(
            predictions, 
            unbinned_labels,
        )

        _, ax = plt.subplots()
        
        plot_prediction_linearity(
            ax,
            unique_labels.detach().cpu().numpy(),
            avgs.detach().cpu().numpy(),
            stds.detach().cpu().numpy(),
            note=(
                f"Event-by-event, {self.level}., "
                + f"{self.num_eval_sets_per_label} boots., "
                + f"{num_events_per_set} events/boots."
                + f"$q^2$ veto: {self.q_squared_veto}"
            ),
        )

        plot_file_name = f"ebe_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_lin.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()
        
        eval_sets_dataset.unload()
    
    def evaluate_sensitivity(self, num_events_per_set):

        eval_sets_dataset = self.single_label_eval_sets_datasets[
            num_events_per_set
        ]
        eval_sets_dataset.load()

        predictions = make_predictions(
            self.model, 
            eval_sets_dataset.features, 
            self.device, 
            event_by_event=True, 
            bin_values=eval_sets_dataset
                .bin_values,
        )
        
        (
            mean, 
            std, 
            bias,
        ) = run_sensitivity_test(
            predictions, 
            self.new_physics_delta_c9_value
        )

        self.summary_table.add_item(self.level, self.q_squared_veto, "Event by event", "Mean at NP", num_events_per_set, mean)
        self.summary_table.add_item(self.level, self.q_squared_veto, "Event by event", "Std. at NP", num_events_per_set, std)
        self.summary_table.add_item(self.level, self.q_squared_veto, "Event by event", "Bias at NP", num_events_per_set, bias)

        _, ax = plt.subplots()

        plot_sensitivity(
            ax, 
            predictions, 
            self.new_physics_delta_c9_value, 
            note=(
                f"Event-by-event, {self.level}., " 
                + f"{self.num_eval_sets_per_label_sensitivity} boots., " 
                + f"{num_events_per_set} events/boots."
                + f"$q^2$ veto: {self.q_squared_veto}"
            ),
        )
        
        plot_file_name = f"ebe_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_sens.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")
        
        plt.show()
        plt.close()

        eval_sets_dataset.unload()