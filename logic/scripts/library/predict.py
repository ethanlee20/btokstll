
import torch
import pandas

from util import get_num_per_unique_label


class Summary_Table:
    def __init__(self):
        self.table = self.make_empty()

    def add_item(
        self, 
        method_name, 
        item_name, 
        num_events_per_set, 
        item,
    ):
        if type(item) is torch.Tensor:
            item = item.item()
        self.table.loc[
            (method_name, num_events_per_set), 
            item_name,
        ] = item
    
    def reset_table(self):
        self.table = self.make_empty()
    
    def make_empty(self):
        index = pandas.MultiIndex.from_product(
            [
                ["gen", "det"],
                [
                    "Images", 
                    "Deep Sets", 
                    "Event by event"    
                ],
                [70_000, 24_000, 6_000],
            ],
            names=["Level", "Method", "Events/set"]
        )
        table = pandas.DataFrame(
            index=index, 
            columns=[
                "MSE",
                "MAE", 
                "Std. at NP", 
                "Mean at NP", 
                "Bias at NP"
            ]
        )
        return table


def make_predictions(
    model, 
    features, 
    device, 
    event_by_event=False, 
    bin_values=None
):
    """
    Make predictions on an array of features.
    """
    with torch.no_grad():
        predictions = []
        for feat in features:
            if event_by_event:
                prediction = model.calculate_expected_value(
                    features.to(device),
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