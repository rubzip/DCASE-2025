from train import train

from models.abstract_model import AbstractModel
from models.classifiers.classifier1 import SimpleDNN
from models.feature_extractors.VGGish200k32v1 import VGGish200k32

from utils.dcase2025_dataset import DCASE2025Dataset
from utils.torch_utils import evaluate_torch
from utils.metrics import macro_class_accuracy_avg


model = AbstractModel(
    embedder=VGGish200k32(),
    classifier=SimpleDNN()
)

train_ds = DCASE2025Dataset.load("data/train.pt")
val_ds = DCASE2025Dataset.load("data/val.pt")

w_train, e_train, y_train = train_ds.mels, train_ds.embeddings, train_ds.scenes
w_val, e_val, y_val = val_ds.mels, val_ds.embeddings, val_ds.scenes

assert len(w_train) == len(e_train) == len(y_train)
assert len(w_val) == len(e_val) == len(y_val)
assert w_val.shape[1:] == w_train.shape[1:]
assert e_val.shape[1:] == e_train.shape[1:]
assert y_val.shape[1:] == y_train.shape[1:]


model = train(model, 
              w_train, e_train, y_train,
              w_val, e_val, y_val)

y_val_p = evaluate_torch(model, w_val)
results = macro_class_accuracy_avg(y_val, y_val_p)
