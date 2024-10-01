
from library.data.datasets.bootstrapped_sets import Bootstrapped_Sets_Dataset

dset = Bootstrapped_Sets_Dataset()

label = "dc9"
n_sig = 42000 # 85_000 max if veto
n_bkg = 42000 # 18_000 max if veto
m_train = 1
m_eval = 1
m_lin_eval = 10

features = ["q_squared", "costheta_mu", "costheta_K", "chi"]
q2_veto = True

sig_data_dir = "../../state/new_physics/data/processed"
bkg_data_dir = "../../state/new_physics/data/processed"
save_dir = "../../state/new_physics/data/processed"

for level in {"gen", "det"}:
    for split, m  in zip(["train", "eval", "lin_eval"], [m_train, m_eval, m_lin_eval]):     
        dset.generate(
            level, split, label,
            n_sig, n_bkg, m, q2_veto,
            sig_data_dir, bkg_data_dir, save_dir
        )

