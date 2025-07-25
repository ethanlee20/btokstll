
from collections import namedtuple

import torch


def predict_values_using_set_based_model(model, features_of_sets_of_events, device):

    def predict_value(model, set_features, device):
        with torch.no_grad():
            predicted_value = model(set_features.to(device).unsqueeze(dim=0))
        return predicted_value
    
    predictions = torch.tensor(
        [
            predict_value(model=model, set_features=set_features, device=device)
            for set_features in features_of_sets_of_events
        ]
    )
    return predictions


def predict_log_probabilities_using_event_based_model(model, features_of_sets_of_events, device):

    def predict_log_probability_distribution(model, set_features, device):
        with torch.no_grad():
            event_logits = model(set_features.to(device))
            event_log_probs = torch.nn.functional.log_softmax(event_logits, dim=1)
            set_logits = torch.sum(event_log_probs, dim=0)
            set_log_probs = torch.nn.functional.log_softmax(set_logits, dim=0)
        return set_log_probs
    
    predictions = torch.cat(
        [
            predict_log_probability_distribution(
                model=model, 
                set_features=set_features, 
                device=device
            ).unsqueeze(dim=0)
            for set_features in features_of_sets_of_events
        ]
    )
    return predictions


def predict_values_using_event_based_model(log_probability_distributions, bin_map, device):

    def calculate_expected_value(log_probability_distribution, bin_map, device):
        with torch.no_grad():
            bin_shift = 5
            log_shifted_bin_map = torch.log(bin_map.to(device) + bin_shift)
            yhat = (
                torch.exp(
                    torch.logsumexp(log_shifted_bin_map + log_probability_distribution.to(device), dim=0)
                ) 
                - bin_shift
            )
        return yhat

    predictions = torch.tensor(
        [
            calculate_expected_value(log_probability_distribution=log_probability_distribution, bin_map=bin_map, device=device)
            for log_probability_distribution in log_probability_distributions
        ]
    )
    return predictions


def run_linearity_test(predicted_values, labels):

    def sort_by_label(predicted_values, labels):
        sorted_labels, sorted_indices = torch.sort(labels)
        sorted_predicted_values = predicted_values[sorted_indices]
        return sorted_predicted_values, sorted_labels
    
    def get_num_sets_per_label(labels):
        unique_labels = torch.unique(labels)
        counts_per_label = torch.tensor(
            [
                len(labels[labels==unique_label])
                for unique_label in unique_labels
            ]
        )
        all_label_counts_are_equal = torch.all(counts_per_label == counts_per_label[0])
        assert all_label_counts_are_equal
        num_sets_per_label = counts_per_label[0]
        return num_sets_per_label

    same_number_of_predictions_and_labels = len(predicted_values) == len(labels)
    assert same_number_of_predictions_and_labels

    with torch.no_grad():
        sorted_predicted_values, sorted_labels = sort_by_label(predicted_values=predicted_values, labels=labels)
        num_sets_per_label = get_num_sets_per_label(labels)
        sorted_predicted_values = sorted_predicted_values.reshape(-1, num_sets_per_label)
        unique_labels = torch.unique(sorted_labels, sorted=False)
        avgs = sorted_predicted_values.mean(dim=1)
        stds = sorted_predicted_values.std(dim=1)
        
    Linearity_Test_Results = namedtuple(
        typename="Linearity_Test_Results", 
        field_names=["unique_labels", "avgs", "stds"]
    )
    return Linearity_Test_Results(unique_labels=unique_labels, avgs=avgs, stds=stds)
    

def run_sensitivity_test(predicted_values, labels):
    
    def get_label(labels):
        unique_labels = torch.unique(labels)
        if len(unique_labels) > 1:
            raise ValueError("Sensitivity test runs on dataset with one label.")
        label = unique_labels.item()
        return label

    label = get_label(labels)
    with torch.no_grad():
        avg = predicted_values.mean()
        std = predicted_values.std()
        bias = avg - label
    
    Sensitivity_Test_Results = namedtuple(
        typename="Sensitivity_Test_Results",
        field_names=["predicted_values", "label", "avg", "std", "bias"]
    )
    return Sensitivity_Test_Results(
        predicted_values=predicted_values, 
        label=label, 
        avg=avg, 
        std=std, 
        bias=bias
    )


def run_error_test(predicted_values, labels):

    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(predicted_values, labels)
        mae = torch.nn.functional.l1_loss(predicted_values, labels)
    
    Error_Test_Results = namedtuple(
        typename="Error_Test_Results",
        field_names=["mse", "mae"]
    )
    return Error_Test_Results(mse=mse, mae=mae)


class Set_Based_Model_Evaluator:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def predict(self, features_of_sets_of_events):
        predictions = predict_values_using_set_based_model(
            model=self.model, 
            features_of_sets_of_events=features_of_sets_of_events, 
            device=self.device
        )
        return predictions

    def run_linearity_test(self, set_dataset):
        results = run_linearity_test(
            predicted_values=self.predict(set_dataset.features), 
            labels=set_dataset.labels
        )
        return results
    
    def run_sensitivity_test(self, sensitivity_set_dataset):
        results = run_sensitivity_test(
            predicted_values=self.predict(sensitivity_set_dataset.features),
            labels=sensitivity_set_dataset.labels
        )
        return results
    
    def run_error_test(self, set_dataset):        
        results = run_error_test(
            predicted_values=self.predict(set_dataset.features), 
            labels=set_dataset.labels
        )
        return results


class Event_Based_Model_Evaluator:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def predict_log_probabilities(self, features_of_sets_of_events):
        predictions = predict_log_probabilities_using_event_based_model(
            model=self.model, 
            features_of_sets_of_events=features_of_sets_of_events, 
            device=self.device
        )
        return predictions

    def predict_values(self, set_dataset):
        predictions = predict_values_using_event_based_model(
            log_probability_distributions=self.predict_log_probabilities(set_dataset.features),
            bin_map=set_dataset.bin_map, # There is only a binned event based model
            device=self.device
        )
        return predictions
    
    def run_linearity_test(self, set_dataset):
        results = run_linearity_test(
            predicted_values=self.predict_values(set_dataset), 
            labels=set_dataset.bin_map[set_dataset.labels]
        )
        return results
    
    def run_sensitivity_test(self, sensitivity_set_dataset):
        results = run_sensitivity_test(
            predicted_values=self.predict_values(sensitivity_set_dataset),
            labels=sensitivity_set_dataset.bin_map[sensitivity_set_dataset.labels]
        )
        return results
    
    def run_error_test(self, set_dataset):
        results = run_error_test(
            predicted_values=self.predict_values(set_dataset), 
            labels=set_dataset.labels
        )
        return results