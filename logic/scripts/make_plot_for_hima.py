
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

path_to_data_charge = Path(r"C:\Users\tetha\Desktop\btokstll\state\bkg_suppression\data\raw\e_gen_charge_sigr.pkl")
path_to_data_mix = Path(r"C:\Users\tetha\Desktop\btokstll\state\bkg_suppression\data\raw\e_gen_mix_sigr.pkl")
path_to_save_dir = Path(r"C:\Users\tetha\Desktop\btokstll\state\bkg_suppression\plots")


df_charge = pd.read_pickle(path_to_data_charge)
df_mix = pd.read_pickle(path_to_data_mix)

df_all = pd.concat([df_charge, df_mix])

df_det = df_all.loc["det"]
df_det_sig = df_det[df_det["isSignal"]==1]

plt.hist(df_det["q_squared"], range=(0,20), bins=100, color="blue", label="all")
plt.hist(df_det_sig["q_squared"], range=(0,20), bins=100, color="red", label="signal")
plt.xlabel("q squared [GeV^2]")
plt.title("B->K*e+e-")
plt.legend()
plt.yscale("log")
plt.savefig(path_to_save_dir.joinpath("q_sq_hist_log.png"), bbox_inches="tight")