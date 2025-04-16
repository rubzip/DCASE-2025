import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchaudio

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_wave(fname: str, secure=False):
    if secure:
        try:
            wave, _ = torchaudio.load(fname)
            return wave
        except:
            return None
    else:
        wave, _ = torchaudio.load(fname)
        return wave


from torch.utils.data import TensorDataset, DataLoader
import torch
def evaluate_torch(fun, tensor, batch_size=64):
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch
            outputs = fun(*inputs)
            all_outputs.append(outputs)

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    return all_outputs_tensor


class EarlyStopping:
    """
    Class for early stopping during training.
    
    Args:
        patience (int): Number of consecutive epochs with no improvement before stopping training.
        verbose (bool): If True, prints messages about the improvement of the monitored metric.
        delta (float): Minimum change in the monitored metric to consider it as an improvement.
        restore_best_weights (bool): If True, stores and restores the model weights when the best metric is achieved.
    """
    def __init__(self, patience=5, verbose=False, delta=0, restore_best_weights=False):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_state = None

    def __call__(self, val_loss, model):
        """
        Calls the EarlyStopping instance with the new validation loss.
        
        Args:
            val_loss (float): Current validation loss value.
            model (torch.nn.Module): The model currently being trained.
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_state = model.state_dict()
            if self.verbose:
                print(f'EarlyStopping initialized. Validation loss: {val_loss:.6f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} of {self.patience} (validation loss: {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping after {self.counter} epochs")
        else:
            self.best_score = score
            if self.restore_best_weights:
                self.best_state = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f'Improvement detected: Validation loss: {val_loss:.6f}. Resetting counter.')

    def restore(self, model):
        """
        Restores the model state to the best saved state.
        
        Args:
            model (torch.nn.Module): The model whose weights will be restored.
        """
        if self.restore_best_weights and self.best_state is not None:
            if self.verbose:
                print("Restoring weights from the best saved model.")
            model.load_state_dict(self.best_state)
        return model

def reduce_dim(train, val, n_components=64, return_torch=True):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    train_norm = scaler.fit_transform(train)
    train_pca = pca.fit_transform(train_norm)

    val_norm = scaler.transform(val)
    val_pca = pca.transform(val_norm)

    if return_torch:
        return torch.from_numpy(train_pca).float(), torch.from_numpy(val_pca).float(), pca.explained_variance_ratio_.sum()
    return train_pca, val_pca, pca.explained_variance_ratio_.sum()
