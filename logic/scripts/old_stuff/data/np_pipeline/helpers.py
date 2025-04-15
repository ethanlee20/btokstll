
"""
Functions for file handling.
"""


import os
from subprocess import run


def image_name(dc9_real, trial):
    proc = run(
        ["bash", "image_name.sh", str(dc9_real), str(trial)],
        text=True,
        capture_output=True,
    )
    result = proc.stdout.replace('\n', '')
    if proc.stderr:
        raise Exception("Problem regarding image name generation.")
    return result


def dec_filename(dc9_real, trial):
    result = f"{image_name(dc9_real, trial)}.dec"
    return result


def mc_filename(dc9_real, trial):
    result = f"{image_name(dc9_real, trial)}.root"
    return result


def make_dec(dc9_real, trial):
    with open('template.dec', 'r') as template:
        text = template.read()
    text = text.replace("DC9REAL", str(dc9_real), 1)

    fname = dec_filename(dc9_real, trial)
    with open(fname, 'x') as dec:
        dec.write(text)
    return fname


def remove_dec(dc9_real, trial):
    os.remove(dec_filename(dc9_real, trial))
    return