import torch
import os
import numpy as np
import sys
from torch import nn, optim
import wandb
from tqdm import tqdm

sys.path.append("/home/xyli/Code/RoboFlamingo")
from student_model.data import load_data
from student_model.model import GPT2FeaturePrediction

config = {
        "num_epochs": 80,
        "ckpt_dir": "/Share/xyli/logs/flamingo/ckpts",
        "vision_dim": 1024,
        "vision_pool_dim": 1536,
        "embed_dim": 2048,
        "feat_dim": 2048,
        "lr": 1e-4,
        "batch_size": 6,
        "exptid": "test_run_gpt",
        "data_size": 100,
        "iterable_data": True,
    }

train_dataloader, val_dataloader = load_data(config)

wandb.init(
    project="student_flamingo",
    name=config["exptid"],
    entity="xinyaoli-sjtu-icec",
    dir="/Share/xyli/logs/flamingo/logs",
)
wandb.config.update(config)

learning_rate = config["lr"]
num_epochs = config["num_epochs"]
ckpt_dir = config["ckpt_dir"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model, loss function, and optimizer
model = GPT2FeaturePrediction(config).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def forward_pass(model, batch, criterion, device):
    text, image, image_pool, target_feature, prev_feature, mask = [t.to(device) for t in batch]
    predicted_feature = model(text, image, image_pool, prev_feature, mask)
    L = target_feature.shape[1]
    predicted_feature = predicted_feature[..., -L:, :]  # (B, L, 2048)
    assert predicted_feature.shape == target_feature.shape, (predicted_feature.shape, target_feature.shape)
    loss = criterion(predicted_feature, target_feature)  # (B, L, 2048)
    return loss

# Training loop remains the same
for epoch in tqdm(range(num_epochs)):
    print(f"\nEpoch {epoch}")

    if epoch % 5 == 0 and epoch > 0:
        with torch.inference_mode():
            model.eval()
            val_loss = 0.0
            for i, batch in enumerate(val_dataloader):
                loss = forward_pass(model, batch, criterion, device)
                val_loss += loss.item()
                if i % 100 == 0:
                    print(f"Val batch {i}, loss: {loss.item():.4f}")
            val_loss /= len(val_dataloader)
        wandb.log({"val/loss": val_loss})
        print(f"Val loss:   {val_loss:.5f}")

    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = forward_pass(model, batch, criterion, device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            wandb.log({"loss": loss.item()})
            print(f"Train batch {i}, loss: {loss.item():.4f}")

    # Average loss for the epoch
    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    ckpt_path = os.path.join(ckpt_dir, config["exptid"], f"model_epoch_{epoch}.ckpt")
    torch.save(model.state_dict(), ckpt_path)

wandb.finish()


