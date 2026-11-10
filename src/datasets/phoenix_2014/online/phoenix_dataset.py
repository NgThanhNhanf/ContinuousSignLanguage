import os
import numpy as np
import pandas as pd
import torch 
import json
from torch.utils.data import Dataset
from datasets.phoenix_2014.utils import read_video

class PhoenixDataset(Dataset):
    def __init__(self, data_path, vocab, window_size = 16):
        with open(data_path) as f:
            self.data = json.load(f)

        self.window_size = window_size
        self.vocab = vocab

        self.data = [
            item for item in self.data
            if self.vocab['state_id_to_gloss_id'][int(item['gloss'])] is not None
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sign_dict = self.data[idx]
        video = read_video(os.path.join('dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner', sign_dict['video_path']))

        random_id_augmentation = np.random.randint(sign_dict['id_start_frame'], sign_dict['id_end_frame'] + 1)
        start = max(0, min(random_id_augmentation - self.window_size // 2, len(video) - self.window_size))
        end = start + self.window_size

        gloss = self.vocab['state_id_to_gloss_id'][int(sign_dict['gloss'])]

        return video[start : end], gloss

class EvaluateDataset(Dataset):
    def __init__(self, data_path, video_dir_path, vocab):
        self.data = pd.read_csv(data_path, delimiter='|')
        self.video_dir_path = video_dir_path
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder_name = self.data['folder'][idx].split('*')[0]
        video_path = os.path.join(self.video_dir_path, folder_name)
        video = read_video(video_path)

        return video, [self.vocab['gloss_to_id'][gloss] for gloss in (" ".join(self.data['annotation'][idx].split(' '))).split(' ')]