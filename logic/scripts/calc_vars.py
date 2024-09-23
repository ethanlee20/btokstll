
from sys import argv

from library.decay_variables import calculate_variables


ell = argv[1]
data_dir = argv[2]

calculate_variables(ell, data_dir, data_dir)