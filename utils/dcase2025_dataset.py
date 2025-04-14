import torch
from torch.utils.data import Dataset

class DCASE2025Dataset(Dataset):
    def __init__(self, filenames: list[str], waves: torch.tensor, mels: torch.tensor, embeddings: torch.tensor, devices: list[str], scenes: torch.tensor):
        assert len(filenames) == len(waves) == len(mels) == len(embeddings) == len(devices) == len(scenes)
        
        self.filenames = filenames
        self.waves = waves
        self.mels = mels
        self.embeddings = embeddings
        self.devices = devices
        self.scenes = scenes

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.filenames[idx], self.waves[idx], self.mels[idx], self.embeddings[idx], self.devices[idx], self.scenes[idx]
    
    def filter_by_device(self, devices: list|str, inplace=False):
        if isinstance(devices, str):
            filtered_indices = [i for i, d in enumerate(self.devices) if d == devices]
        else:
            filtered_indices = [i for i, d in enumerate(self.devices) if d in devices]
        filenames_filtered = [self.filenames[i] for i in filtered_indices]
        devices_filtered = [self.devices[i] for i in filtered_indices]

        waves_filtered = self.waves[filtered_indices]
        mels_filtered = self.mels[filtered_indices]
        embeddings_filtered = self.embeddings[filtered_indices]
        scenes_filtered = self.scenes[filtered_indices]

        if inplace:
            self.filenames = filenames_filtered
            self.devices_filtered = devices_filtered

            self.waves_filtered = waves_filtered
            self.mels_filtered = mels_filtered
            self.embeddings_filtered = embeddings_filtered
            self.scenes_filtered = scenes_filtered

            return
        
        return DCASE2025Dataset(filenames_filtered, waves_filtered, mels_filtered, embeddings_filtered, devices_filtered, scenes_filtered)
    
    def save(self, path):
        data_to_save = {
            "filenames": self.filenames,
            "waves": self.waves,
            "mels": self.mels,
            "embeddings": self.embeddings,
            "devices": self.devices,
            "scenes": self.scenes
        }
        torch.save(data_to_save, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path)
        return cls(
            filenames=data["filenames"],
            waves=data["waves"],
            mels=data["mels"],
            embeddings=data["embeddings"],
            devices=data["devices"],
            scenes=data["scenes"]
        )