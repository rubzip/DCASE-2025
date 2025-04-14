import torch
from torch.utils.data import DataLoader, TensorDataset

from hear21passt.base import load_model, get_scene_embeddings

def get_embeddings(waves, batch_size=64):
    model = load_model()

    dataset = TensorDataset(waves)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            embs = get_scene_embeddings(inputs, model)
            all_embs.append(embs)

    all_outputs_tensor = torch.cat(all_embs, dim=0)
    return all_outputs_tensor
