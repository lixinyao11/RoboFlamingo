import torch
import os
import numpy as np
import h5py
from torch import nn, optim
from transformers import GPT2Model
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


config = {
        "num_epochs": 100,
        "ckpt_dir": "student_model/ckpts",
        "vision_dim": 1024,
        "vision_pool_dim": 1536,
        "embed_dim": 2048,
        "feat_dim": 2048,
        "lr": 1e-4,
        "batch_size": 6,
        "exptid": "test_run",
        "data_file_num": 50,
    }


class MyDataset(Dataset):
    def __init__(self, dir):
        # read all hdf5 files in the directory
        self.hdf5_files = [f for f in os.listdir(dir) if f.endswith('.hdf5')]
        self.text = []
        self.image = []
        self.image_pool = []
        self.features = []
        self.time_steps = []
        cnt = 0
        for file in self.hdf5_files:
            if cnt > config["data_file_num"]:
                break
            cnt += 1
            print("Reading file", cnt, file)
            with h5py.File(os.path.join(dir, file), 'r') as f:
                for group in f.keys():
                    self.text.append(np.array(f[group]['text']))  # (T, 1, L, 2048)
                    self.image.append(np.array(f[group]['image']))  # (T, 1, 1, 1, 256+256, 1024)
                    self.image_pool.append(np.array(f[group]['image.pooled']))  # (T, 1, 768+768)
                    self.features.append(np.array(f[group]['features']))  # (T, 1, L, 2048)
                    self.time_steps.append(f[group]['text'].shape[0])  # T

    def __len__(self):
        total_length = np.sum(np.array(self.time_steps))
        return int(total_length)

    def __getitem__(self, idx):
        # Find the appropriate group and sample within it
        cumulative_idx = 0
        for i in range(len(self.time_steps)):
            if idx < cumulative_idx + self.time_steps[i]:
                local_idx = idx - cumulative_idx
                text = torch.tensor(self.text[i][local_idx], dtype=torch.float32)[0]  # (L, 2048)
                image = torch.tensor(self.image[i][local_idx], dtype=torch.float32)[0][0][0]  # (256+256, 1024)
                image_pool = torch.tensor(self.image_pool[i][local_idx], dtype=torch.float32)[0] # (768+768)
                feature = torch.tensor(self.features[i][local_idx], dtype=torch.float32)[0]  # (L, 2048)
                prev_feature = torch.tensor(self.features[i][local_idx - 1], dtype=torch.float32)[0] if local_idx > 0 else torch.zeros_like(feature)
                text_pool = torch.mean(text, dim=0)  # (2048,)
                prev_feature_pool = torch.mean(prev_feature, dim=0)  # (2048,)
                return text, text_pool, image, image_pool, feature, prev_feature, prev_feature_pool
            cumulative_idx += self.time_steps[i]

def collate_fn(batch):
    texts, texts_pool, images, images_pool, features, prev_features, prev_features_pool = zip(*batch)
    max_length = max(max(t.size(0) for t in texts), max(f.size(0) for f in features), max(pf.size(0) for pf in prev_features))

    def pad_to_max_length(tensor_list, max_length):
        padded_list = []
        for tensor in tensor_list:
            padding_size = max_length - tensor.size(0)
            if padding_size > 0:
                padding = torch.zeros((padding_size, *tensor.shape[1:]), dtype=tensor.dtype)
                padded_tensor = torch.cat([tensor, padding], dim=0)
            else:
                padded_tensor = tensor
            padded_list.append(padded_tensor)
        return torch.stack(padded_list, dim=0)
    
    texts = pad_to_max_length([t.clone().detach() for t in texts], max_length)
    features = pad_to_max_length([f.clone().detach() for f in features], max_length)
    prev_features = pad_to_max_length([f.clone().detach() for f in prev_features], max_length)

    B, L, _ = texts.shape
    assert features.shape[0] == B
    assert prev_features.shape[0] == B
    assert features.shape[1] == L
    assert prev_features.shape[1] == L

    mask = torch.ones(B, 1 + L + 1 + 512 + 1 + L)
    mask[:, 1:1+L] = (texts.sum(dim=-1) != 0).float()
    mask[:, -L:] = (prev_features.sum(dim=-1) != 0).float()

    texts_pool = torch.stack(texts_pool, dim=0)
    texts = torch.cat([texts_pool.unsqueeze(1), texts], dim=1)
    prev_features_pool = torch.stack(prev_features_pool, dim=0)
    prev_features = torch.cat([prev_features_pool.unsqueeze(1), prev_features], dim=1)
    
    images = torch.stack(images, dim=0)
    images_pool = torch.stack(images_pool, dim=0).unsqueeze(1)

    # (B, L+1, 2048), (B, 512, 1024), (B, 1, 1536), (B, L, 2048), (B, L+1, 2048), (B, 1+L+1+512+1+L)
    return texts, images, images_pool, features, prev_features, mask


BATCH_SIZE = config["batch_size"]
VISION_DIM = config["vision_dim"]
VISION_POOL_DIM = config["vision_pool_dim"]
EMBED_DIM = config["embed_dim"]
FEAT_DIM = config["feat_dim"]

# Create the dataset and dataloader
data_dir = '/Share/xyli/Datasets/flamingo_data/logs_20241002_011951'
dataset = MyDataset(data_dir)
print(f"Total number of timesteps: {len(dataset)}")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn, 
    num_workers=4,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

wandb.init(
    project="student_flamingo",
    name=config["exptid"],
    entity="xinyaoli-sjtu-icec",
    dir="student_model/logs",
)
wandb.config.update(config)

class GPT2FeaturePrediction(nn.Module):
    def __init__(self):
        super(GPT2FeaturePrediction, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        hidden_dim = self.gpt2.config.hidden_size
        self.project_text = nn.Linear(EMBED_DIM, hidden_dim)
        self.project_feature = nn.Linear(FEAT_DIM, hidden_dim)
        self.project_vision = nn.Linear(VISION_DIM, hidden_dim)
        self.project_vision_pool = nn.Linear(VISION_POOL_DIM, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, FEAT_DIM)

    def forward(self, text, image, image_pool, prev_feature, mask):
        text = self.project_text(text)
        image = self.project_vision(image)
        image_pool = self.project_vision_pool(image_pool)
        prev_feature = self.project_feature(prev_feature)
        combined_input = torch.cat([text, image_pool, image, prev_feature], dim=1)  # (B, seq_L, hidden_dim)
        gpt2_output = self.gpt2(inputs_embeds=combined_input, attention_mask=mask).last_hidden_state
        output_feature = self.linear_out(gpt2_output.squeeze(0))
        return output_feature


learning_rate = config["lr"]
num_epochs = config["num_epochs"]
ckpt_dir = config["ckpt_dir"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model, loss function, and optimizer
model = GPT2FeaturePrediction().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def forward_pass(model, batch, criterion, device):
    text, image, image_pool, target_feature, prev_feature, mask = [t.to(device) for t in batch]
    predicted_feature = model(text, image, image_pool, prev_feature, mask)
    L = target_feature.shape[1]
    predicted_feature = predicted_feature[..., -L:, :]  # (B, L, 2048)
    assert predicted_feature.shape == target_feature.shape
    loss = criterion(predicted_feature, target_feature)  # (B, L, 2048)
    return loss

# Training loop remains the same
for epoch in tqdm(range(num_epochs)):
    print(f"\nEpoch {epoch}")

    if epoch % 2 == 0:
        with torch.inference_mode():
            model.eval()
            val_loss = 0.0
            for i, batch in enumerate(val_dataloader):
                loss = forward_pass(model, batch, criterion, device)
                val_loss += loss.item()
                # if i > 50:
                #     break
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Val loss:   {val_loss:.5f}")

    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = forward_pass(model, batch, criterion, device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch})
            print(f"Train loss: {loss.item():.4f}")

    # Average loss for the epoch
    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    ckpt_path = os.path.join(ckpt_dir, config["exptid"], f"model_epoch_{epoch}.ckpt")
    torch.save(model.state_dict(), ckpt_path)

wandb.finish()


