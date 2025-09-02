
from collections import namedtuple

from numpy import sqrt, histogram, linspace


def calculate_efficiency(generator_level_series, detector_level_series, num_bins, bounds):
    
    """
    Calculate the efficiency per bin.

    The efficiency of bin i is defined as the number of
    detector entries in i divided by the number of generator
    entries in i.

    The error for bin i is calculated as the squareroot of the
    number of detector entries in i divided by the number of
    generator entries in i.
    """

    def make_bins(bounds, num_bins):
        assert len(bounds) == 2
        bin_edges, bin_width = linspace(start=bounds[0], stop=bounds[1], num=num_bins+1, retstep=True)
        bin_middles = linspace(start=bounds[0]+bin_width/2, stop=bounds[1]-bin_width/2, num=num_bins)
        return bin_edges, bin_middles

    bin_edges, bin_middles = make_bins(bounds, num_bins)

    generator_level_histogram, _ = histogram(generator_level_series, bins=bin_edges)
    detector_level_histogram, _ = histogram(detector_level_series, bins=bin_edges)

    print("num events generator: ", generator_level_histogram.sum())
    print("num events detector: ", detector_level_histogram.sum())

    efficiencies = detector_level_histogram / generator_level_histogram
    errors = sqrt(detector_level_histogram) / generator_level_histogram

    Efficiency_Result = namedtuple('Efficiency_Result', ['bin_middles', 'efficiencies', 'errors'])
    
    result = Efficiency_Result(bin_middles=bin_middles, efficiencies=efficiencies, errors=errors)
    return result