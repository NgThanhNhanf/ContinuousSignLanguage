import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence

from src.model.cslr.cslr_model import CNNTransformer
from src.datasets.phoenix_2014.offline.phoenix_dataset import PhoenixDataset
from src.datasets.phoenix_2014.utils import build_gloss_vocabulary
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAME = 300
NUM_CLASSES = 1232
WEIGHTS = torchvision.models.ResNet18_Weights.DEFAULT   
EPOCHS = 40

train_videos_dir = 'dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train'
train_data_path = 'dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv'
val_videos_dir = 'dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev'
val_data_path = 'dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv'
gloss_to_id_path = 'gloss_to_id.json'

model = CNNTransformer(n_frames=NUM_FRAME, 
                       num_classes=NUM_CLASSES, 
                       weights=WEIGHTS).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
criterion = torch.nn.CTCLoss(
    blank=1231,
    reduction='mean',
    zero_infinity=True
)

train_dataset = PhoenixDataset(train_videos_dir, train_data_path, gloss_to_id_path)
val_dataset = PhoenixDataset(val_videos_dir, val_data_path, gloss_to_id_path, True)

def collate_fn(batch):
    videos, glosses = zip(*batch)

    # Original sequence lengths
    lengths = torch.tensor(
        [video.size(0) for video in videos],
        dtype=torch.long
    )

    # Pad videos to the longest sequence in this batch
    videos = pad_sequence(
        videos,
        batch_first=True,
        padding_value=0.0
    )

    # (B, T_max)
    T_max = videos.size(1)

    # True means padding position
    padding_mask = (
        torch.arange(T_max).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )

    # CTC targets
    target_lengths = torch.tensor(
        [len(gloss) for gloss in glosses],
        dtype=torch.long
    )

    targets = torch.cat(glosses)

    return (
        videos,
        lengths,
        padding_mask,
        targets,
        target_lengths
    )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = 1,
                                           shuffle = True,
                                           collate_fn = collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size = 1,
                                           shuffle = False, 
                                           collate_fn = collate_fn)

for epoch in range(EPOCHS):
    # model.train()
    # train_loss = 0.0

    # for videos, lengths, padding_mask, glosses, glosses_lengths in tqdm(train_loader, desc = 'Training: '):
    #     videos = videos.to(device)
    #     glosses = glosses.to(device)
    #     optimizer.zero_grad()

    #     logits, log_probs = model(videos, padding_mask)
    #     log_probs = log_probs.transpose(0, 1)

    #     loss = criterion(
    #         log_probs,
    #         glosses,
    #         lengths,
    #         glosses_lengths
    #     )
    #     loss.backward()
    #     optimizer.step()

    #     train_loss += loss.item()

    # train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.inference_mode():
        for videos, lengths, padding_mask, glosses, glosses_lengths in tqdm(val_loader, desc = 'Valid: '):
            videos = videos.to(device)
            glosses = glosses.to(device)
    
            logits, log_probs = model(videos, padding_mask)
            log_probs = log_probs.transpose(0, 1)

    
            loss = criterion(
                log_probs,
                glosses,
                lengths,
                glosses_lengths
            )
    
            val_loss += loss.item()

        val_loss /= len(train_loader)

    print(f'Epoch {epoch + 1}/{EPOCHS}: Train loss: {train_loss} | Val loss: {val_loss}')
        