import torch
import h5py
import os
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset

class MyDataset(Dataset):
    def __init__(self, dir, split='train', split_ratio=0.9, data_cut=None):
        self.data_dir = dir
        self.files = [f for f in os.listdir(dir) if f.endswith('.hdf5')]
        if data_cut:
            self.files = self.files[:data_cut]
        split_idx = int(len(self.files) * split_ratio)
        if split == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        self.file_meta = []
        for i, file in enumerate(self.files):
            print("Reading file", i, file)
            with h5py.File(os.path.join(self.data_dir, file), 'r') as f:
                for group in f.keys():
                    self.file_meta.append({"id": i, "subtask": group, "length": f[group]['text'].shape[0]})

    def __len__(self):
        total_length = sum(meta["length"] for meta in self.file_meta)
        return int(total_length)

    def __getitem__(self, idx):
        # Find the appropriate group and sample within it
        cumulative_idx = 0
        for meta in self.file_meta:
            if idx < cumulative_idx + meta["length"]:
                local_idx = idx - cumulative_idx
                with h5py.File(os.path.join(self.data_dir, self.files[meta["id"]]), 'r') as f:
                    group = meta["subtask"]
                    text = torch.tensor(f[group]['text'][local_idx, 0, ...], dtype=torch.float32)  # (L, 2048)
                    image = torch.tensor(f[group]['image'][local_idx, 0, 0, 0, ...], dtype=torch.float32)  # (256+256, 1024)
                    image_pool = torch.tensor(f[group]['image.pooled'][local_idx, 0, ...], dtype=torch.float32)  # (768+768)
                    feature = torch.tensor(f[group]['features'][local_idx, 0, ...], dtype=torch.float32)  # (L, 2048)
                    prev_feature = torch.tensor(f[group]['features'][local_idx - 1, 0, ...], dtype=torch.float32) if local_idx > 0 else torch.zeros_like(feature)

                    text_pool = torch.mean(text, dim=0)  # (2048,)
                    prev_feature_pool = torch.mean(prev_feature, dim=0)  # (2048,)
                    return text, text_pool, image, image_pool, feature, prev_feature, prev_feature_pool
            cumulative_idx += meta["length"]


class MyIterableDataset(IterableDataset):
    def __init__(self, dir, split='train', split_ratio=0.9, data_size=None, chunk_size=30):
        self.data_dir = dir
        self.files = [f for f in os.listdir(dir) if f.endswith('.hdf5')]
        if data_size:
            self.files = self.files[:data_size]
        split_idx = int(len(self.files) * split_ratio)
        if split == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]
        self.current_file_index = 0
        self.current_data = []
        self.chunk_size = chunk_size
        self.len = 0
        for file in self.files:
            with h5py.File(os.path.join(self.data_dir, file), 'r') as f:
                for group in f.keys():
                    self.len += f[group]['text'].shape[0]

    def __len__(self):
        return self.len

    def load_next_chunk(self):
        for i in range(self.chunk_size):
            if self.current_file_index < len(self.files):
                file_path = self.files[self.current_file_index]
                with h5py.File(os.path.join(self.data_dir, file_path), 'r') as f:
                    for group in f.keys():
                        self.current_data.append({
                            "subtask": group,
                            "length": f[group]['text'].shape[0],
                            "text": f[group]['text'][:],
                            "image": f[group]['image'][:],
                            "image.pooled": f[group]['image.pooled'][:],
                            "features": f[group]['features'][:],
                        })
                self.current_file_index += 1
            else:
                break

    def __iter__(self):
        return self.iter_data()
    
    def iter_data(self):
        self.current_data = []
        self.current_file_index = 0
        while self.current_file_index < len(self.files) or self.current_data:
            if not self.current_data:
                self.load_next_chunk()
            for group in self.current_data:
                for t in range(group["length"]):
                    text = torch.tensor(group["text"][t, 0, ...], dtype=torch.float32)  # (L, 2048)
                    image = torch.tensor(group["image"][t, 0, 0, 0, ...], dtype=torch.float32)  # (256+256, 1024)
                    image_pool = torch.tensor(group["image.pooled"][t, 0, ...], dtype=torch.float32)  # (768+768)
                    feature = torch.tensor(group["features"][t, 0, ...], dtype=torch.float32)  # (L, 2048)
                    if t > 0:
                        prev_feature = torch.tensor(group["features"][t - 1, 0, ...], dtype=torch.float32) 
                    else:
                        prev_feature = torch.zeros_like(feature)
                    text_pool = torch.mean(text, dim=0)  # (2048,)
                    prev_feature_pool = torch.mean(prev_feature, dim=0)  # (2048,)
                    yield text, text_pool, image, image_pool, feature, prev_feature, prev_feature_pool
            self.current_data = []


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


def load_data(config):
    data_dir = '/Share/xyli/Datasets/flamingo_data/logs_20241002_011951'
    if config["iterable_data"]:
        train_dataset = MyIterableDataset(data_dir, split='train', data_size=config["data_size"])
        val_dataset = MyIterableDataset(data_dir, split='val', data_size=config["data_size"])
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=0,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
    else:
        train_dataset = MyDataset(data_dir, split='train', data_cut=config["data_size"])
        val_dataset = MyDataset(data_dir, split='val', data_cut=config["data_size"])
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn, 
            num_workers=4,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

    return train_dataloader, val_dataloader