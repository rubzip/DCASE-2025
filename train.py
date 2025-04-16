import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models.abstract_model import AbstractModel
from utils.metrics import CombinedLoss, macro_class_accuracy_avg
from utils.torch_utils import EarlyStopping


def train(
        model: AbstractModel,
        w_train, e_train, y_train,
        w_val, e_val, y_val,
        alpha_e=1.,
        alpha_y=1.,
        e_loss_fn=None,
        y_loss_fn=None,
        lr=0.001,
        l2=0.0001,
        batch_size=32,
        patience=10,
        num_epochs=100,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=True):
    
    criterion = CombinedLoss(alpha_e=alpha_e, alpha_y=alpha_y, e_loss_fn=e_loss_fn, y_loss_fn=y_loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    early_stopper = EarlyStopping(patience=patience, verbose=verbose, restore_best_weights=True)

    train_dataset = TensorDataset(w_train, e_train, y_train)
    val_dataset = TensorDataset(w_val, e_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        all_train_labels = []
        all_train_preds = []    

        for batch in train_loader:
            inputs, embeddings, labels = batch
            
            optimizer.zero_grad()
            embeddings_pred, outputs = model(inputs)
            loss = criterion(embeddings_pred, embeddings, outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            all_train_labels.append(labels)
            all_train_preds.append(predicted)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = macro_class_accuracy_avg(all_train_labels, all_train_preds) * 100

        val_loss = 0.
        model.eval()
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, embeddings, labels = batch
                embeddings_pred, outputs = model(inputs)
                loss = criterion(embeddings_pred, embeddings, outputs, labels)

                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                all_val_labels.append(labels)
                all_val_preds.append(predicted)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = macro_class_accuracy_avg(all_val_labels, all_val_preds) * 100
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% || "
                f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            break
    
    model = early_stopper.restore(model)
    return model
