
import pickle


class Loss_Table:
    """
    Loss table.
    """
    
    def __init__(self):
        self.epochs = []
        self.losses_train = []
        self.losses_eval = []

    def append(
        self, 
        epoch, 
        loss_train, 
        loss_eval
    ):
        self.epochs.append(epoch)
        self.losses_train.append(loss_train)
        self.losses_eval.append(loss_eval)
    
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(
                self, 
                handle, 
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def load(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
            self.epochs = data.epochs
            self.losses_train = data.losses_train
            self.losses_eval = data.losses_eval
            


    