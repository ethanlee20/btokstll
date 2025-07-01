

import torch

from .model import Custom_Model
from ..data.dset.dataset import Custom_Dataset
from .constants import Names_Models
from ..data.dset.preproc import values_from_probs


class Evaluator:

    def __init__(
        self,
        model:Custom_Model,
        dataset:Custom_Dataset,
        device:str,
    ):

        self.model = model

        self.dataset = dataset

        self.device = device

        self._send_model_to_device()

        self.labels = (
            values_from_probs(
                self.dataset.labels,
                self.dataset.bin_map,
            )
            if model.config.name == Names_Models().ebe
            else self.dataset.labels 
        )

        if (
            model.config.name 
            == Names_Models().ebe
        ):
            
            self._send_bin_map_to_device()

    def predict(self):

        def predict_ebe(x):

            log_probs = (
                self.model.model
                .predict_log_probs(x)
            )

            pred = (
                self.model.model
                .calculate_expected_value(
                    log_probs, 
                    self.dataset.bin_map,
                )
            )

            return pred, log_probs

        with torch.no_grad():

            preds = []
            log_probs = []
            
            for x in self.dataset.features:

                x = x.to(self.device)
                
                if (
                    self.model.config.name 
                    == Names_Models().ebe
                ):

                    pred, log_probs_ = predict_ebe(x=x)
                
                    log_probs.append(log_probs_.unsqueeze(dim=0))

                else:

                    pred = self.model(
                        x.unsqueeze(0)
                    )
                
                preds.append(pred)
            
            self.preds = torch.tensor(preds)

            if self.model.config.name == Names_Models().ebe:
                self.log_probs = torch.cat(log_probs)
    
    def run_test_lin(self):

        def sort(preds, labels):

            labels_sorted, ind_sorted = torch.sort(
                labels
            )

            preds_sorted = preds[ind_sorted]

            return preds_sorted, labels_sorted

        with torch.no_grad():

            preds, labels = sort(
                self.preds,
                self.labels
            )

            preds = preds.reshape(
                -1,
                self.dataset
                .config
                .num_sets_per_label
            )

            labels_unique = torch.unique(
                labels
            )

            avgs = preds.mean(dim=1)
            stds = preds.std(dim=1)

            return labels_unique, avgs, stds
        
    def run_test_sens(self):

        def get_label(labels):

            labels_unique = torch.unique(
                labels
            )

            if len(
                labels_unique
            ) > 1:
                
                raise ValueError(
                    "Sensitivity test runs on "
                    "dataset with one label."
                )
            
            label = labels_unique.item()

            return label

        label = get_label(
            self.labels
        )

        avg = self.preds.mean()

        std = self.preds.std()

        bias = avg - label

        return avg, std, bias, label
    
    def calc_mse_mae(self):

        with torch.no_grad():

            mse = (
                torch.nn.functional
                .mse_loss(
                    self.preds, 
                    self.labels
                )
            )

            mae = (
                torch.nn.functional
                .l1_loss(
                    self.preds, 
                    self.labels
                )
            )

        return mse, mae

    def _send_model_to_device(self):

        self.model = self.model.to(
            self.device
        )

    def _send_bin_map_to_device(self):

        self.dataset.bin_map = (
            self.dataset
            .bin_map
            .to(
                self.device
            )
        )

