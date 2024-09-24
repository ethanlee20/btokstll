
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

path_to_data_charge = Path(r"C:\Users\tetha\Desktop\btokstll\state\new_physics\data\raw\bkg\mu_sideb_generic_charge_calc.pkl")
path_to_data_mix = Path(r"C:\Users\tetha\Desktop\btokstll\state\new_physics\data\raw\bkg\mu_sideb_generic_mix_calc.pkl")
path_to_save_dir = Path(r"C:\Users\tetha\Desktop\btokstll\state\new_physics\plots")


df_charge = pd.read_pickle(path_to_data_charge)
df_mix = pd.read_pickle(path_to_data_mix)

df_all = pd.concat([df_charge, df_mix])

df_det = df_all.loc["det"]

plt.hist(df_det["q_squared"], range=(0,20), bins=100)
plt.xlabel("q squared [GeV^2]")
plt.title("B->K*mu+mu-")
plt.savefig(path_to_save_dir.joinpath("q_sq_hist.png"), bbox_inches="tight")