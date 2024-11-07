
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import log_softmax, softmax

from library.modeling.models.deep_sets_cat import Deep_Sets_Cat
from library.data.datasets.bootstrapped_sets_binned import Bootstrapped_Sets_Binned_Dataset


level="gen"

dset = Bootstrapped_Sets_Binned_Dataset()
dset.load(level, "eval", "../../state/new_physics/data/processed")

bin = 10

set_index = np.argwhere(dset.labels==bin)[0]

input_data = dset.sets[set_index]

model_file_path = "../../state/new_physics/models/deep_sets_gen_cat.pt"

model = Deep_Sets_Cat()

model.load_state_dict(torch.load(model_file_path, weights_only=True))

model.eval()
with torch.no_grad():
    
    model_output = model(input_data) # ok

    log_prob = log_softmax(model_output, dim=1) # ok

    prob = softmax(model_output, dim=1)

    for i in range(prob.shape[0]):
        plt.plot(prob[i])
    plt.savefig("../../state/new_physics/plots/cat_results.png")