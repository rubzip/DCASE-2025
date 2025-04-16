import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder


class AbstractModel(nn.Module):
    def __init__(self, embedder, classifier, preprocessor=None, name=None):
        super().__init__()
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()
        self.embedder = embedder
        self.classifier = classifier

        self.name = f"{type(embedder).__name__} + {type(classifier).__name__}" if name is None else name
        self.encoder = LabelEncoder().fit([
            "airport", "bus", "metro", "metro_station", "park", "public_square", "shopping_mall", "street_pedestrian", "street_traffic", "tram"
            ])

    def forward(self, w, return_embedding=True):
        x = self.preprocessor(w)
        e = self.embedder(x)
        y = self.classifier(e)

        if return_embedding:
            return e, y
        return y

    def predict(self, w):
        y = self.forward(w, return_embedding=False)
        indices = torch.argmax(y, dim=1)
        return self.encoder.inverse_transform(indices)
