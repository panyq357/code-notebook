from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from tqdm import tqdm

import config
from model import MyModel
from dataloader import get_train_dataset
from dataloader import get_volcabulary


model_file = Path(config.MODEL_FILE)
outdir = model_file.parent

if not outdir.exists():
    outdir.mkdir(parents=True)


dataloader = DataLoader(
    Subset(get_train_dataset(), list(range(config.SUBSET_TRAIN))),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
)

volcabulary = get_volcabulary()


if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = MyModel(vocab_size=len(volcabulary)).to(device)

model.train()

loss_fn = CrossEntropyLoss()

optimizer = Adam(params=model.parameters(), lr=config.LEARNING_RATE)


best_loss = float("inf")
for epoch in range(config.EPOCH_NUM):

    epoch_total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):

        input = batch["input"].to(device)
        target = batch["target"].to(device)

        output = model(input)
        
        loss = loss_fn(output, target)
        epoch_total_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    epoch_avg_loss = epoch_total_loss / len(dataloader)
    print("Epoch", epoch, "average loss:", epoch_avg_loss)
    if epoch_avg_loss < best_loss:
        best_loss = epoch_avg_loss
        torch.save(model.state_dict(), model_file)


