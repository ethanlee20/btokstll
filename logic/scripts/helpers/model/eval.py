

import pandas
import torch





def make_predictions(
    model, 
    features, 
    device, 
    event_by_event=False, 
    bin_values=None
):
    """
    Make predictions on an array of features.

    Features should be an array of sets of events.
    Bin values must be specified for event-by-event method.
    """
    with torch.no_grad():
        predictions = []
        for feat in features:
            if event_by_event:
                assert bin_values is not None
                prediction = model.calculate_expected_value(
                    feat.to(device),
                    bin_values.to(device),
                )
            else:
                prediction = model(feat.unsqueeze(0).to(device))
            predictions.append(prediction)
        predictions = torch.tensor(predictions)
    return predictions


def run_linearity_test(predictions, labels):
    """
    Calculate the average and standard deviation of
    predictions for each label.

    DANGER: Assumes data sorted by labels.
    
    Returns
    -------
    unique_labels : ...
    avg_yhat_per_label : ...
    std_yhat_per_label : ...
    """

    with torch.no_grad():
        num_sets_per_label = get_num_per_unique_label(labels)
        avg_yhat_per_label = predictions.reshape(-1, num_sets_per_label).mean(dim=1)
        std_yhat_per_label = predictions.reshape(-1, num_sets_per_label).std(dim=1)
        unique_labels = torch.unique(labels)
    return unique_labels, avg_yhat_per_label, std_yhat_per_label


def run_sensitivity_test(predictions, label):
    """
    Find the standard deviation and mean of predictions for
    a single label.

    Returns
    -------
    mean : ...
    std : ...
    bias : ...
    """
    mean = predictions.mean()
    std = predictions.std()
    bias = mean - label
    return mean, std, bias


def calculate_mse_mae(predictions, labels):
    """
    Calculate the mean squared error and the mean absolute error.

    Returns
    -------
    mse : ...
    mae : ...
    """
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(predictions, labels)
        mae = torch.nn.functional.l1_loss(predictions, labels)
    return mse, mae


def predict_log_probabilities(self, x):
    """
    Predict the log probability of each class, given a set of events.

    x : A torch tensor of features of events. (a set)
    """
    with torch.no_grad():
        event_logits = self.forward(x)
        event_log_probabilities = torch.nn.functional.log_softmax(event_logits, dim=1)
        set_logits = torch.sum(event_log_probabilities, dim=0)
        set_log_probabilities = torch.nn.functional.log_softmax(set_logits, dim=0)
    return set_log_probabilities

def calculate_expected_value(self, x, bin_values):
    """
    Calculate the prediction expectation value, given a set of events.

    x : A torch tensor of features of events. (a set)
    """
    with torch.no_grad():
        bin_shift = 5
        bin_values = bin_values + bin_shift
        log_bin_values = torch.log(bin_values)
        log_probs = self.predict_log_probabilities(x)
        lse = torch.logsumexp(log_bin_values + log_probs, dim=0)
        yhat = torch.exp(lse) - bin_shift
    return yhat