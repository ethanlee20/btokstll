
from pathlib import Path
import pandas as pd

def make_file_name(kind, split):
    assert kind in {"charge", "mix"}
    assert split in {"train", "eval"}
    name = f"bkg_{kind}_{split}.pkl"
    return name


class Background_Dataset():
    def __init__(self):
        pass
    
    def generate(
        self,
        features,
        bkg_charge_all_file_path, bkg_mix_all_file_path,
        save_dir,
        veto_q2=True,
        dtype='float32'
    ):
        save_dir = Path(save_dir)
        
        file_names = {
            "charge_train": make_file_name("charge", "train"),
            "charge_eval": make_file_name("charge", "eval"),
            "mix_train": make_file_name("mix", "train"),
            "mix_eval": make_file_name("mix", "eval"),
        }
        
        save_file_paths = {k: save_dir.joinpath(v) for k, v in file_names.items()}

        df_bkg_charge_all = pd.read_pickle(bkg_charge_all_file_path).loc["det"][features]
        df_bkg_mix_all = pd.read_pickle(bkg_mix_all_file_path).loc["det"][features]

        if veto_q2:
            apply_q2_cut = lambda x: x[(x["q_squared"]>1) & (x["q_squared"]<8)]
            df_bkg_charge_all = apply_q2_cut(df_bkg_charge_all)
            df_bkg_mix_all = apply_q2_cut(df_bkg_mix_all)

        df_bkg_charge_all = df_bkg_charge_all.astype(dtype)
        df_bkg_mix_all = df_bkg_mix_all.astype(dtype)

        ns = {
            "charge_train": int(len(df_bkg_charge_all)/2),
            "charge_eval": len(df_bkg_charge_all) - int(len(df_bkg_charge_all)/2),
            "mix_train": int(len(df_bkg_mix_all)/2),
            "mix_eval": len(df_bkg_mix_all) - int(len(df_bkg_mix_all)/2)
        }

        bkg_dfs = {
            "charge_train": 
                df_bkg_charge_all.sample(
                    n=ns["charge_train"], replace=False
                ),
            "charge_eval": 
                df_bkg_charge_all.sample(
                    n=ns["charge_eval"], replace=False
                ),
            "mix_train": 
                df_bkg_mix_all.sample(
                    n=ns["mix_train"], replace=False
                ),
            "mix_eval": 
                df_bkg_mix_all.sample(
                    n=ns["mix_eval"], replace=False
                )
        }

        for key in bkg_dfs:
            bkg_dfs[key].to_pickle(save_file_paths[key])

    def load(self, split, save_dir):
        save_dir = Path(save_dir)
        file_names = {
            "charge": make_file_name("charge", split),
            "mix": make_file_name("mix", split),
        }
        file_paths = {k: save_dir.joinpath(v) for k, v in file_names.items()}

        self.df_charge = pd.read_pickle(file_paths["charge"])
        self.df_mix = pd.read_pickle(file_paths["mix"])

