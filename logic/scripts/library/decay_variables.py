
from pathlib import Path

import numpy as np
import pandas as pd

from .math_calculations import (
    cosine_angle, 
    cross_product_3d, 
    dot_product, 
    square_matrix_transform, 
    unit_normal, 
    vector_magnitude,
)
from .utilities.file_handling import open_data


def four_momemtum_dataframe(df_with_4_col):
    """
    Create a four-momentum dataframe.

    Create a dataframe where each row represents a four-momentum.
    The columns are well labeled.
    The returned dataframe is a new dataframe.
    """

    df_4mom = df_with_4_col.copy()
    df_4mom.columns = ["E", "px", "py", "pz"]
    return df_4mom


def three_momemtum_dataframe(df_with_3_col):
    """
    Create a three-momentum dataframe.
    
    Create a dataframe where each row represents a three-momentum.
    The columns are well labeled.
    The returned dataframe is a new dataframe.
    """

    df_3mom = df_with_3_col.copy()
    df_3mom.columns = ["px", "py", "pz"]
    return df_3mom


def three_velocity_dataframe(df_with_3_col):
    """
    Create a three-velocity dataframe.
    
    Create a dataframe where each row represents a three-velocity.
    The columns are well labeled.
    The returned dataframe is a new dataframe.
    """
    
    df_3vel = df_with_3_col.copy()
    df_3vel.columns = ["vx", "vy", "vz"]
    return df_3vel


def inv_mass_sq_two_particles(
    df_p1_4mom, df_p2_4mom
):
    """
    Compute the squares of the invariant masses for two particles systems.

    Given the four-momentum dataframe of particle 1 and of particle 2,
    return an invariant masses squared series.
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

    df_invM_sq = df_sum_E**2 - df_sum_3mom_mag_sq
    return df_invM_sq


def three_velocity_from_four_momentum_dataframe(df_4mom):
    """
    Compute a three-velocity dataframe from a four-momentum dataframe.
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

    Given a dataframe of three velocities,
    return a series of corresponding Lorentz factors.
    """

    df_3vel = three_velocity_dataframe(df_3vel)
    df_vel_mag = vector_magnitude(df_3vel)
    df_gamma = 1 / np.sqrt(1 - df_vel_mag**2)

    return df_gamma


def compute_Lorentz_boost_matrix(df_vel3vec):
    """
    Compute a dataframe of Lorentz boost matricies.

    Given a three velocity dataframe, compute the
    corresponding dataframe of Lorentz boost matricies.
    Each row contains a different matrix.
    """

    df_vel3vec = df_vel3vec.copy()
    df_vel3vec.columns = ["vx", "vy", "vz"]
    df_vel_mag = vector_magnitude(df_vel3vec)
    df_gamma = compute_gamma(df_vel3vec)

    df_boost_matrix = pd.DataFrame(
        data=np.zeros(shape=(df_vel3vec.shape[0], 16)),
        index=df_vel3vec.index,
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
    df_boost_matrix["b01"] = -df_gamma * df_vel3vec["vx"]
    df_boost_matrix["b02"] = -df_gamma * df_vel3vec["vy"]
    df_boost_matrix["b03"] = -df_gamma * df_vel3vec["vz"]
    df_boost_matrix["b10"] = -df_gamma * df_vel3vec["vx"]
    df_boost_matrix["b11"] = (
        1
        + (df_gamma - 1)
        * df_vel3vec["vx"] ** 2
        / df_vel_mag**2
    )
    df_boost_matrix["b12"] = (
        (df_gamma - 1)
        * df_vel3vec["vx"]
        * df_vel3vec["vy"]
        / df_vel_mag**2
    )
    df_boost_matrix["b13"] = (
        (df_gamma - 1)
        * df_vel3vec["vx"]
        * df_vel3vec["vz"]
        / df_vel_mag**2
    )
    df_boost_matrix["b20"] = -df_gamma * df_vel3vec["vy"]
    df_boost_matrix["b21"] = (
        (df_gamma - 1)
        * df_vel3vec["vy"]
        * df_vel3vec["vx"]
        / df_vel_mag**2
    )
    df_boost_matrix["b22"] = (
        1
        + (df_gamma - 1)
        * df_vel3vec["vy"] ** 2
        / df_vel_mag**2
    )
    df_boost_matrix["b23"] = (
        (df_gamma - 1)
        * df_vel3vec["vy"]
        * df_vel3vec["vz"]
        / df_vel_mag**2
    )
    df_boost_matrix["b30"] = -df_gamma * df_vel3vec["vz"]
    df_boost_matrix["b31"] = (
        (df_gamma - 1)
        * df_vel3vec["vz"]
        * df_vel3vec["vx"]
        / df_vel_mag**2
    )
    df_boost_matrix["b32"] = (
        (df_gamma - 1)
        * df_vel3vec["vz"]
        * df_vel3vec["vy"]
        / df_vel_mag**2
    )
    df_boost_matrix["b33"] = (
        1
        + (df_gamma - 1)
        * df_vel3vec["vz"] ** 2
        / df_vel_mag**2
    )

    return df_boost_matrix


def boost(df_ref_4mom, df_4vec):
    """
    Lorentz boost a 4-vector.

    Lorentz boost a 4-vector to a frame
    given by a reference 4-momentum vector.
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


def find_costheta_ell(df_ell_p_4mom, df_ell_m_4mom, df_B_4mom):
    """
    Find the cosine of the muon helicity angle for B -> K* ell+ ell-.
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

    df_costheta_ell = cosine_angle(
        df_ellell_3mom_Bframe, df_ell_p_3mom_ellellframe
    )

    return df_costheta_ell


def find_costheta_K(df_K_4mom, df_KST_4mom, df_B_4mom):
    """
    Find the cosine of the K* helicity angle for B -> K* ell+ ell-.
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

    df_costheta_K = cosine_angle(
        df_KST_3mom_Bframe, df_K_3mom_KSTframe
    )

    return df_costheta_K


def find_unit_normal_KST_K_plane(
    df_B_4mom, df_KST_4mom, df_K_4mom
):
    """
    Find the unit normal to the plane made by the
    direction vectors of the K* and K.

    This is for B -> K* ell+ ell-.
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
    the positively charged lepton.

    This is for B -> K* ell+ ell-.
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

    df_unit_normal_ellell_ellplus_plane = unit_normal(
        df_ell_p_3mom_ellellframe, df_ellell_3mom_Bframe
    )
    return df_unit_normal_ellell_ellplus_plane


def find_coschi(
    df_B_4mom,
    df_K_4mom,
    df_KST_4mom,
    df_ell_p_4mom,
    df_ell_m_4mom,
):
    """
    Find the cosine of the decay angle chi.

    Chi is the angle between the K* K decay plane and the dilepton ell+ decay plane.

    This is for B -> K* ell+ ell-.
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

    coschi = dot_product(
        df_unit_normal_KST_K_plane,
        df_unit_normal_ellell_ellplus_plane,
    )

    return coschi


def find_chi(
    df_B_4mom,
    df_K_4mom,
    df_KST_4mom,
    df_ell_p_4mom,
    df_ell_m_4mom,
):
    """
    Find the decay angle chi.

    Chi is the angle between the K* K decay plane and the dilepton ell+ decay plane.
    It can range from 0 to 2*pi.

    This is for B -> K* ell+ ell-.
    """

    coschi = find_coschi(
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

    n_ell_cross_n_K = cross_product_3d(
        df_unit_normal_ellell_ellplus_plane,
        df_unit_normal_KST_K_plane,
    )

    df_B_4mom = four_momemtum_dataframe(df_B_4mom)
    df_KST_4mom = four_momemtum_dataframe(df_KST_4mom)
    df_KST_4mom_Bframe = boost(
        df_ref_4mom=df_B_4mom, df_4vec=df_KST_4mom
    )
    df_KST_3mom_Bframe = three_momemtum_dataframe(
        df_KST_4mom_Bframe[["px", "py", "pz"]]
    )

    n_ell_cross_n_K_dot_Kst = dot_product(
        n_ell_cross_n_K, df_KST_3mom_Bframe
    )
    chi = np.sign(n_ell_cross_n_K_dot_Kst) * np.arccos(
        coschi
    )

    def to_positive_angles(chi):
        return chi.where(chi > 0, chi + 2 * np.pi)

    chi = to_positive_angles(chi)

    return chi


def calc_dif_inv_mass_k_pi_and_kst(df_K_4mom, df_pi_4mom):
    """
    Calcualate the difference between the invariant mass of the K pi system
    and the K*'s invariant mass (PDG value).
    """

    inv_mass_kst = 0.892

    df_inv_mass_k_pi = np.sqrt(
        inv_mass_sq_two_particles(df_K_4mom, df_pi_4mom)
    )

    dif = df_inv_mass_k_pi - inv_mass_kst
    return dif


def calculate_variables(ell, in_path, out_dir):
    """
    Calculate decay variables from reconstructed/generator data.

    Parameters
    ----------
    ell : str
        "mu" or "e"
    in_path : str
        Path to a data file or data directory.
        If the path is to a directory, all files
        in the directory will be loaded (recursively).
    out_dir : str
        The directory to which the output will be saved.

    Side Effects
    ------------
    - Saves the result to the specified directory.
    """
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_file_name = f"{in_path.stem}_calc.pkl"
    out_file_path = out_dir.joinpath(out_file_name)

    if ell == 'mu':
        ell_p_E = 'mu_p_E'
        ell_p_px = 'mu_p_px'
        ell_p_py = 'mu_p_py'
        ell_p_pz = 'mu_p_pz'
        ell_p_mcE = 'mu_p_mcE'
        ell_p_mcPX = 'mu_p_mcPX'
        ell_p_mcPY = 'mu_p_mcPY'
        ell_p_mcPZ = 'mu_p_mcPZ'
        ell_m_E = 'mu_m_E'
        ell_m_px = 'mu_m_px'
        ell_m_py = 'mu_m_py'
        ell_m_pz = 'mu_m_pz'
        ell_m_mcE = 'mu_m_mcE'
        ell_m_mcPX = 'mu_m_mcPX'
        ell_m_mcPY = 'mu_m_mcPY'
        ell_m_mcPZ = 'mu_m_mcPZ'
        costheta_ell = 'costheta_mu'
        costheta_ell_mc = 'costheta_mu_mc'
    elif ell == 'e':
        ell_p_E = 'e_p_E'
        ell_p_px = 'e_p_px'
        ell_p_py = 'e_p_py'
        ell_p_pz = 'e_p_pz'
        ell_p_mcE = 'e_p_mcE'
        ell_p_mcPX = 'e_p_mcPX'
        ell_p_mcPY = 'e_p_mcPY'
        ell_p_mcPZ = 'e_p_mcPZ'
        ell_m_E = 'e_m_E'
        ell_m_px = 'e_m_px'
        ell_m_py = 'e_m_py'
        ell_m_pz = 'e_m_pz'
        ell_m_mcE = 'e_m_mcE'
        ell_m_mcPX = 'e_m_mcPX'
        ell_m_mcPY = 'e_m_mcPY'
        ell_m_mcPZ = 'e_m_mcPZ'
        costheta_ell = 'costheta_e'
        costheta_ell_mc = 'costheta_e_mc'
    else:
        raise ValueError(f"ell not recognized: {ell}")

    df = open_data(in_path)

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

    df["q_squared"] = inv_mass_sq_two_particles(
        df_ell_p_4mom, df_ell_m_4mom
    )
    df["q_squared_mc"] = inv_mass_sq_two_particles(
        df_ell_p_4mom_mc, df_ell_m_4mom_mc
    )
    df[costheta_ell] = find_costheta_ell(
        df_ell_p_4mom, df_ell_m_4mom, df_B_4mom
    )
    df[costheta_ell_mc] = find_costheta_ell(
        df_ell_p_4mom_mc, df_ell_m_4mom_mc, df_B_4mom_mc
    )
    df["costheta_K"] = find_costheta_K(
        df_K_4mom, df_KST_4mom, df_B_4mom
    )
    df["costheta_K_mc"] = find_costheta_K(
        df_K_4mom_mc, df_KST_4mom_mc, df_B_4mom_mc
    )
    df["coschi"] = find_coschi(
        df_B_4mom,
        df_K_4mom,
        df_KST_4mom,
        df_ell_p_4mom,
        df_ell_m_4mom,
    )
    df["coschi_mc"] = find_coschi(
        df_B_4mom_mc,
        df_K_4mom_mc,
        df_KST_4mom_mc,
        df_ell_p_4mom_mc,
        df_ell_m_4mom_mc,
    )
    df["chi"] = find_chi(
        df_B_4mom,
        df_K_4mom,
        df_KST_4mom,
        df_ell_p_4mom,
        df_ell_m_4mom,
    )
    df["chi_mc"] = find_chi(
        df_B_4mom_mc,
        df_K_4mom_mc,
        df_KST_4mom_mc,
        df_ell_p_4mom_mc,
        df_ell_m_4mom_mc,
    )
    df["invM_K_pi_shifted"] = calc_dif_inv_mass_k_pi_and_kst(
        df_K_4mom,
        df_pi_4mom
    )
    df["invM_K_pi_shifted_mc"] = calc_dif_inv_mass_k_pi_and_kst(
        df_K_4mom_mc,
        df_pi_4mom_mc
    )

    df.to_pickle(out_file_path)




