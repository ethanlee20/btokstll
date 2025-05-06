
from .physics import (
    four_momemtum_dataframe,
    inv_mass_sq_two_particles,
    find_costheta_ell,
    find_costheta_K,
    find_chi,
    find_coschi,
    calc_dif_inv_mass_k_pi_and_kst
)


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

