
from numpy import pi

from ..datasets.constants import Names_of_Variables
from ..datasets.make_and_save.preprocessing import apply_q_squared_veto


def calculate_resolution(detector_level_dataframe, name_of_variable):

    """
    Calculate the resolution.

    The resolution of a variable is defined as the
    reconstructed value minus the MC truth value.

    If the variable is chi, periodicity is accounted for.
    """

    def apply_periodicity(resolution):
        resolution = resolution.where(resolution < pi, resolution - 2 * pi)
        resolution = resolution.where(resolution > -pi, resolution + 2 * pi)
        return resolution
    
    detector_level_dataframe = apply_q_squared_veto(detector_level_dataframe, "loose")
    
    measured = detector_level_dataframe[name_of_variable]
    generated = detector_level_dataframe[name_of_variable+'_mc']
    resolution = measured - generated

    if name_of_variable == Names_of_Variables().chi:
        resolution = apply_periodicity(resolution)

    return resolution