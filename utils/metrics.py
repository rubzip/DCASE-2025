import torch
from torch import nn

def macro_class_accuracy_avg(y_true, y_pred, return_classwise=False):
    """
    Computes macro-averaged class-wise accuracy in PyTorch.

    Args:
        y_true (torch.Tensor): 1D tensor of ground truth labels (int64).
        y_pred (torch.Tensor): 1D tensor of predicted labels (int64).

    Returns:
        float: macro-averaged accuracy.
        dict: per-class accuracies (as floats).
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    
    classes = torch.unique(y_true)
    accuracies = {}

    for cls in classes:
        # Mask to select all samples of the current class
        cls_mask = (y_true == cls)
        total = cls_mask.sum().item()

        if total == 0:
            acc = 0.0
        else:
            # Count correct predictions for this class
            correct = (y_pred[cls_mask] == y_true[cls_mask]).sum().item()
            acc = correct / total

        # Store accuracy for the current class
        accuracies[int(cls.item())] = acc

    if return_classwise:
        return accuracies
    
    # Compute macro-average of all per-class accuracies
    macro_avg = sum(accuracies.values()) / len(accuracies)

    return macro_avg


class CombinedLoss(nn.Module):
    def __init__(self, alpha_e=1.0, alpha_y=1.0, e_loss_fn=None, y_loss_fn=None):
        """
        Initializes a combined loss function that includes:
          - A loss between predicted and target embeddings (e_pred vs e_theor).
          - A classification loss between predicted logits and true labels (y_pred vs y_theor).

        Args:
            alpha_e (float): Weight for the embedding loss component.
            alpha_y (float): Weight for the classification loss component.
            e_loss_fn (callable, optional): Loss function for embeddings (default: nn.MSELoss).
            y_loss_fn (callable, optional): Loss function for classification (default: nn.CrossEntropyLoss).
        """
        super().__init__()
        self.alpha_e = alpha_e
        self.alpha_y = alpha_y
        self.e_loss_fn = nn.MSELoss() if e_loss_fn is None else e_loss_fn
        self.y_loss_fn = nn.CrossEntropyLoss if y_loss_fn is None else y_loss_fn

    def forward(self, e_pred, e_theor, y_pred, y_theor):
        """
        Computes the total loss as a weighted sum of:
          - Embedding loss (e_pred vs e_theor)
          - Classification loss (y_pred vs y_theor)

        Args:
            e_pred (Tensor): Predicted embeddings.
            e_theor (Tensor): Target embeddings.
            y_pred (Tensor): Logits predicted by the model.
            y_theor (Tensor): True class labels.

        Returns:
            Tensor: Combined loss value.
        """
        loss_e = self.e_loss_fn(e_pred, e_theor)
        loss_y = self.y_loss_fn(y_pred, y_theor)
        total_loss = self.alpha_e * loss_e + self.alpha_y * loss_y
        return total_loss
