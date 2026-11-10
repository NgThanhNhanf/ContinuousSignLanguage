import torch
import torchvision

from src.model.islr.islr_model import CNNTransformer
from datasets.phoenix_2014.online.phoenix_dataset import PhoenixDataset, EvaluateDataset
from datasets.phoenix_2014.utils import build_gloss_vocabulary
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAME = 16
NUM_CLASSES = 3694
WEIGHTS = torchvision.models.ResNet18_Weights.DEFAULT   
EPOCHS = 40

train_path = 'train_dictionary.json'
val_data_path = 'dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv'
val_video_dir_path = 'dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev'
model = CNNTransformer(n_frames=NUM_FRAME, 
                       num_classes=NUM_CLASSES, 
                       weights=WEIGHTS).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
criterion = torch.nn.CrossEntropyLoss()

vocab = build_gloss_vocabulary(
    "dataset/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/annotations/automatic/trainingClasses.txt"
)

# print(len(vocab['state_id_to_gloss_id']))
print(vocab['gloss_to_id'].keys())
# print(vocab['state_id_to_gloss_id'][3693])

train_dataset = PhoenixDataset(train_path, vocab)
val_dataset = EvaluateDataset(val_data_path, val_video_dir_path, vocab)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = 8,
                                           shuffle = True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size = 8,
                                           shuffle = False)

for epoch in range(EPOCHS):
    # model.train()
    # train_loss = 0.0

    # for videos, glosses in tqdm(train_loader, desc = 'Training: '):
    #     videos = videos.to(device)
    #     glosses = glosses.to(device)
    #     optimizer.zero_grad()

    #     output = model(videos)

    #     loss = criterion(output, glosses)
    #     loss.backward()
    #     optimizer.step()

    #     train_loss += loss.item()

    # train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.inference_mode():
        for videos, glosses in tqdm(val_loader, desc = 'Valid: '):
            videos = videos.to(device)
            glosses = glosses.to(device)
    
            output = model(videos)
    
            loss = criterion(output, glosses)
    
            val_loss += loss.item()

        val_loss /= len(train_loader)

    print(f'Epoch {epoch + 1}/{EPOCHS}: Train loss: {train_loss} | Val loss: {val_loss}')
        