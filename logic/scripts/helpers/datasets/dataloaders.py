
from numpy import floor
import torch


def get_num_batches_per_epoch(dataset_length, batch_size):
    num_batches_per_epoch = int(floor(dataset_length/batch_size))
    return num_batches_per_epoch


def get_epoch_indices(dataset_length, num_batches, batch_size):
    num_examples = num_batches * batch_size
    indices = torch.randperm(dataset_length)[:num_examples]
    batched_indices = torch.reshape(indices, shape=(num_batches, batch_size))
    return batched_indices


def get_batch_indices(epoch_indices, batch_index):
    batch_indices = epoch_indices[batch_index]
    return batch_indices


def get_features_batch(features, batch_indices):
    features_batch = features[batch_indices]
    return features_batch


def get_labels_batch(labels, batch_indices):
    labels_batch = labels[batch_indices]
    return labels_batch


class Custom_Data_Loader:

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last,
        shuffle,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        if not self.drop_last: raise NotImplementedError
        if not self.shuffle: raise NotImplementedError

        self.num_batches_per_epoch = get_num_batches_per_epoch(
            dataset_length=len(dataset), 
            batch_size=batch_size
        )
    
    def __len__(self):
        return self.num_batches_per_epoch
    
    def __iter__(self):
        self.current_batch_index = 0
        self.epoch_indices = get_epoch_indices(
            dataset_length=len(self.dataset),
            num_batches=self.num_batches_per_epoch,
            batch_size=self.batch_size
        )
        return self
    
    def __next__(self):
        if self.current_batch_index >= self.num_batches_per_epoch:
            raise StopIteration
        batch_indices = get_batch_indices(
            epoch_indices=self.epoch_indices,
            batch_index=self.current_batch_index
        )
        features = get_features_batch(
            features=self.dataset.features,
            batch_indices=batch_indices
        )
        labels = get_labels_batch(
            labels=self.dataset.labels,
            batch_indices=batch_indices
        )
        self.current_batch_index += 1
        return features, labels
