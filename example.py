import torch

from models.abstract_model import AbstractModel

from models.feature_extractors.VGGish200k32v1 import VGGish200k32
from models.preprocessing.mel_features import LogMelSpectrogram
from models.classifiers.classifier1 import SimpleDNN

from utils.complexity import get_model_size_bytes, get_torch_macs_memory


def get_model_stats(embedder, classifier, preprocessor=None, sr=44_100):
    if preprocessor is None:
        model = AbstractModel(
            embedder=embedder,
            classifier=classifier
        )
        w = torch.zeros((1, 44_100))
    else:
        model = AbstractModel(
            preprocessor=preprocessor,
            embedder=embedder,
            classifier=classifier
        )
        w = torch.zeros((1, 44_100))
    
    data = []




model = AbstractModel(
    embedder=VGGish200k32(),
    classifier=SimpleDNN(),
    preprocessor=LogMelSpectrogram()
)


embedder = VGGish200k32()
classifier = SimpleDNN()
preprocessor = LogMelSpectrogram()

w = torch.zeros((1, 44_100))
x = preprocessor(w)
#e = embedder(x)
print(f"Preprocessor:\n  Size: {get_model_size_bytes(preprocessor)} Bytes\n  MACs: {get_torch_macs_memory(preprocessor, w.shape)}")
print(f"Embedder:\n  Size: {get_model_size_bytes(embedder)} Bytes\n  MACs: {get_torch_macs_memory(embedder, x.shape)[0]}")




print(get_torch_macs_memory(model, input_size=w.shape))

#print(get_torch_macs_memory(model, ))

