
from library.data.datasets.bootstrapped_sets import Bootstrapped_Sets_Dataset

dset = Bootstrapped_Sets_Dataset()

label = "dc9"
n_sig = 18_000 # 85_000 max if veto
n_bkg = 18_000 # 18_000 max if veto
m = 1

features = ["q_squared", "costheta_mu", "costheta_K", "chi"]
q2_veto = True

sig_data_dir = "../../state/new_physics/data/processed"
save_dir = "../../state/new_physics/data/processed"
bkg_charge_file = "../../state/new_physics/data/raw/bkg/mu_sideb_generic_charge_calc.pkl"
bkg_mix_file = "../../state/new_physics/data/raw/bkg/mu_sideb_generic_mix_calc.pkl"

for level in {"gen", "det"}:
    for split in {"train", "eval"}:     
        dset.generate(
            level, split, features, label,
            n_sig, n_bkg, m, q2_veto,
            sig_data_dir, bkg_charge_file, bkg_mix_file, save_dir
        )

