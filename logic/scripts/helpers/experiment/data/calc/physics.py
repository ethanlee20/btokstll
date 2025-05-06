"""
Physics
"""

import numpy
import pandas

from .df_math import (
    vector_magnitude,
    square_matrix_transform,
    dot_product,
    cosine_angle,
    cross_product_3d,
    unit_normal
)

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
