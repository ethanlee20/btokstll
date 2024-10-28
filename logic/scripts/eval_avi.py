
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import log_softmax, softmax

from library.modeling.models.single_event_nn import Single_Event_NN
from library.data.datasets.aggregated_signal_binned import Aggregated_Signal_Binned_Dataset


level="gen"

dset = Aggregated_Signal_Binned_Dataset()
dset.load(level, "train", "../../state/new_physics/data/processed")

bin = np.argwhere(dset.bins==-2).item()

input_data = dset.feat[np.argwhere(dset.labels==bin).squeeze()]#[0:10]

model_file_path = "../../state/new_physics/models/single_event_nn_gen.pt"

model = Single_Event_NN()

model.load_state_dict(torch.load(model_file_path, weights_only=True))

model.eval()
with torch.no_grad():
    
    model_output = model(input_data) # ok

    log_prob_event = log_softmax(model_output, dim=1) # ok

    sum_log_prob_event = torch.sum(log_prob_event, 0) # ok

    n_bins = dset.bins.shape[0] # ok

    prior_prob = 1/n_bins # ok

    n_events = input_data.shape[0] # ok

    sum_log_prob_prior = (n_events-1) * torch.log(torch.Tensor([prior_prob])) # ok

    log_prob = sum_log_prob_event - sum_log_prob_prior

    prob = torch.exp(log_prob)

    plt.plot(dset.bins, log_prob)
    plt.savefig("../../state/new_physics/plots/avi_results.png")

    # breakpoint()