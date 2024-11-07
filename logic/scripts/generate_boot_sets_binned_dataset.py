
from library.data.datasets.bootstrapped_sets_binned import Bootstrapped_Sets_Binned_Dataset

dset = Bootstrapped_Sets_Binned_Dataset()

label = "dc9"
n_sig = 24_000 # 85_000 max if veto
n_bkg = 4 # 18_000 max if veto
m_train = 30
m_eval = 5
m_lin_eval = 30

features = ["q_squared", "costheta_mu", "costheta_K", "chi"]
q2_veto = True

sig_data_dir = "../../state/new_physics/data/processed"
bkg_data_dir = "../../state/new_physics/data/processed"
save_dir = "../../state/new_physics/data/processed"

for level in {"gen", "det"}:
    for split, m  in zip(["train", "eval", "lin_eval"], [m_train, m_eval, m_lin_eval]):     
        dset.generate(
            level, split,
            n_sig, n_bkg, m,
            sig_data_dir, bkg_data_dir, save_dir
        )

