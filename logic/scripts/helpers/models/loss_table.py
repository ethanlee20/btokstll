
import pickle


class Loss_Table:

    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.eval_losses = []

    def append(self, epoch, train_loss, eval_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.eval_losses.append(eval_loss)
    
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
            self.epochs = data.epochs
            self.train_losses = data.train_losses
            self.eval_losses = data.eval_losses
            


    