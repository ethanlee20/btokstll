

from library.util import aggregate_raw_signal


level="gen"

feature_names = [ # DANGER: Changing this order could break things.
    "q_squared", 
    "costheta_mu", 
    "costheta_K", 
    "chi"
]

dataset_save_dir = "../../state/new_physics/data/processed"
raw_signal_dir = "../../state/new_physics/data/raw/signal"

# Training split
aggregate_raw_signal(
    level, 
    raw_trials=range(1,21), 
    columns=feature_names, 
    raw_signal_dir=raw_signal_dir, 
    save_dir=dataset_save_dir,
)

# Evaluation split
aggregate_raw_signal(
    level,
    raw_trials=range(21,41),
    columns=feature_names,
    raw_signal_dir=raw_signal_dir,
    save_dir=dataset_save_dir,
)