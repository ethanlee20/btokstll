
from library.data.datasets.background import Background_Dataset


dset = Background_Dataset()

features = ["q_squared", "costheta_mu", "costheta_K", "chi"]

save_dir = "../../state/new_physics/data/processed"
bkg_charge_file = "../../state/new_physics/data/raw/bkg/mu_sideb_generic_charge_calc.pkl"
bkg_mix_file = "../../state/new_physics/data/raw/bkg/mu_sideb_generic_mix_calc.pkl"

dset.generate(
    features,
    bkg_charge_file, bkg_mix_file, 
    save_dir
)

