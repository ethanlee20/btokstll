#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Steering file for new physics MC generation.
"""


import sys
from pathlib import Path

import basf2 as b2
import simulation as si
import reconstruction as re
import mdst as mdst
import glob as glob

from helpers import make_dec, remove_dec, mc_filename


dc9_real = float(sys.argv[1])
trial = int(sys.argv[2])
n_events = int(sys.argv[3])
output_dir = Path(sys.argv[4])

path_dec = make_dec(dc9_real, trial)
path_file_out = output_dir.joinpath(mc_filename(dc9_real, trial))

print("\n-- Input Information --")
print("dc9_real: ", dc9_real)
print("trial: ", trial)
print("n_events: ", n_events)
print("dec file: ", path_dec)
print("sim out file: ", path_file_out)
print("-----------------------\n")

# background (collision) files
bg = glob.glob('/group/belle2/dataprod/BGOverlay/early_phase3/release-06-00-05/overlay/BGx1/set0/*.root')
# background if running locally
bg_local = glob.glob('/group/belle2/dataprod/BGOverlay/early_phase3/release-06-00-05/overlay/BGx1/set0/*.root')

# set database conditions (in addition to default)
b2.conditions.prepend_globaltag("mc_production_MC15ri_a")

# create path
main = b2.Path()

# default to early phase 3 (exp=1003), run 0
main.add_module("EventInfoSetter", expList=1003, runList=0, evtNumList=n_events)

# generate events from decfile
main.add_module('EvtGenInput', userDECFile=b2.find_file(path_dec))

# detector simulation
si.add_simulation(path=main, bkgfiles=bg)

# reconstruction
re.add_reconstruction(path=main)

# Finally add mdst output (file name overwritten on the grid)
mdst.add_mdst_output(path=main, filename=str(path_file_out))

# process events and print call statistics
b2.process(path=main)
print(b2.statistics)

remove_dec(dc9_real, trial)