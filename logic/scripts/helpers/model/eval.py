

import torch

from .model import Custom_Model
from ..data.dset.dataset import Custom_Dataset
from .constants import Names_Models


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

        if (
            model.config.name 
            == Names_Models().ebe
        ):
            self._send_bin_map_to_device()

    def predict(self, x):

        def predict_ebe(x):

            log_probs = predict_log_probs_ebe(
                self.model, 
                x
            )
            pred = calculate_expected_value_ebe(
                log_probs, 
                self.dataset.bin_map,
            )

            return pred

        with torch.no_grad():

            preds = []
            
            for x_ in x:

                x_ = x_.to(self.device)
                
                if (
                    self.model.config.name 
                    == Names_Models().ebe
                ):

                    pred = predict_ebe(x_)
                
                else:

                    pred = self.model(
                        x_
                        .unsqueeze(0)
                    )
                
                preds.append(pred)
            
            self.preds = torch.tensor(preds)
    
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
                self.dataset.labels
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
            self.dataset.labels
        )

        avg = self.preds.mean()
        
        std = self.preds.std()
        
        bias = avg - label

        return avg, std, bias
    
    def calc_mse_mae(self):

        with torch.no_grad():

            mse = (
                torch.nn.functional
                .mse_loss(
                    self.preds, 
                    self.dataset.labels
                )
            )

            mae = (
                torch.nn.functional
                .l1_loss(
                    self.preds, 
                    self.dataset.labels
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
        



def predict_log_probs_ebe(
    model_ebe, 
    x
):
    """
    Predict the log probability 
    of each class, given a set of events.

    x : torch.Tensor 
        Features of events. (a set)
    """

    with torch.no_grad():
        
        event_logits = (
            model_ebe
            .forward(x)
        )
       
        event_log_probs = (
            torch.nn.functional
            .log_softmax(
                event_logits, 
                dim=1,
            )
        )

        set_logits = torch.sum(
            event_log_probs, 
            dim=0,
        )
        
        set_log_probs = (
            torch.nn.functional
            .log_softmax(
                set_logits, 
                dim=0
            )
        )

    return set_log_probs


def calculate_expected_value_ebe(
    log_probs,
    bin_map
):
    """
    Calculate the prediction expectation 
    value, given log probabilities.

    log_probs : torch.Tensor 
        Log probabilities.
        Output of predict_log_proba_ebe.
    """

    with torch.no_grad():
    
        bin_shift = 5
    
        bin_map = bin_map + bin_shift
    
        log_bin_map = torch.log(bin_map)
    
        lse = torch.logsumexp(
            log_bin_map + log_probs, 
            dim=0
        )
    
        yhat = torch.exp(lse) - bin_shift
    
    return yhat


# def make_predictions(
#     model, 
#     features, 
#     device, 
#     event_by_event=False, 
#     bin_values=None
# ):
#     """
#     Make predictions on an array of features.

#     Features should be an array of sets of events.
#     Bin values must be specified for event-by-event method.
#     """
#     with torch.no_grad():
#         predictions = []
#         for feat in features:
#             if event_by_event:
#                 assert bin_values is not None
#                 prediction = model.calculate_expected_value(
#                     feat.to(device),
#                     bin_values.to(device),
#                 )
#             else:
#                 prediction = model(feat.unsqueeze(0).to(device))
#             predictions.append(prediction)
#         predictions = torch.tensor(predictions)
#     return predictions


# def run_linearity_test(predictions, labels):
#     """
#     Calculate the average and standard deviation of
#     predictions for each label.

#     DANGER: Assumes data sorted by labels.
    
#     Returns
#     -------
#     unique_labels : ...
#     avg_yhat_per_label : ...
#     std_yhat_per_label : ...
#     """

#     with torch.no_grad():
#         num_sets_per_label = get_num_per_unique_label(labels)
#         avg_yhat_per_label = predictions.reshape(-1, num_sets_per_label).mean(dim=1)
#         std_yhat_per_label = predictions.reshape(-1, num_sets_per_label).std(dim=1)
#         unique_labels = torch.unique(labels)
#     return unique_labels, avg_yhat_per_label, std_yhat_per_label


# def run_sensitivity_test(predictions, label):
#     """
#     Find the standard deviation and mean of predictions for
#     a single label.

#     Returns
#     -------
#     mean : ...
#     std : ...
#     bias : ...
#     """
#     mean = predictions.mean()
#     std = predictions.std()
#     bias = mean - label
#     return mean, std, bias


# def calculate_mse_mae(predictions, labels):
#     """
#     Calculate the mean squared error and the mean absolute error.

#     Returns
#     -------
#     mse : ...
#     mae : ...
#     """
#     with torch.no_grad():
#         mse = torch.nn.functional.mse_loss(predictions, labels)
#         mae = torch.nn.functional.l1_loss(predictions, labels)
#     return mse, mae