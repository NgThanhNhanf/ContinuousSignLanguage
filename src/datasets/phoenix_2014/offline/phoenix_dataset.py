import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.datasets.phoenix_2014.utils import read_video
import json

class PhoenixDataset(Dataset):
    def __init__(self, video_dir, data_path, gloss_to_id_path, is_ctc = False):
        self.data = pd.read_csv(data_path, delimiter='|')
        with open(gloss_to_id_path) as f:
            self.gloss_to_id = json.load(f)
        if is_ctc:
            self.data = self.data[
                ~self.data["annotation"].apply(
                    lambda x: self.has_oov(x)
                )
            ].reset_index(drop=True)
        self.video_dir = video_dir


    def has_oov(self, annotation):
        glosses = annotation.strip().split()

        return any(
            gloss not in self.gloss_to_id
            for gloss in glosses
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, (self.data['folder'][idx]).split('*')[0])

        video = read_video(video_path)

        annotation = self.data['annotation'][idx]

        gloss_tokens = annotation.strip().split()

        glosses = [
            self.gloss_to_id[gloss]
            for gloss in gloss_tokens
        ]
        glosses = torch.from_numpy(np.array(glosses))

        return video, glosses