import math
import torch
import h5py
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset

class MyDataset(Dataset):
    def __init__(self, dir, norm_stats, split='train', split_ratio=0.9, data_size=None):
        self.data_dir = dir
        self.norm_stats = norm_stats
        self.files = [f for f in os.listdir(dir) if f.endswith('.hdf5')]
        if data_size:
            self.files = self.files[:data_size]
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
                    feature = (feature - self.norm_stats["feature_mean"]) / self.norm_stats["feature_std"]
                    prev_feature = torch.tensor(f[group]['features'][local_idx - 1, 0, ...], dtype=torch.float32) if local_idx > 0 else torch.zeros_like(feature)
                    prev_feature = (prev_feature - self.norm_stats["feature_mean"]) / self.norm_stats["feature_std"]

                    text_pool = torch.mean(text, dim=0)  # (2048,)
                    prev_feature_pool = torch.mean(prev_feature, dim=0)  # (2048,)
                    return text, text_pool, image, image_pool, feature, prev_feature, prev_feature_pool
            cumulative_idx += meta["length"]


class MyIterableDataset(IterableDataset):
    def __init__(self, dir, norm_stats, split='train', split_ratio=0.9, data_size=None):
        self.data_dir = dir
        self.norm_stats = norm_stats
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
        self.len = 0
        for file in self.files:
            with h5py.File(os.path.join(self.data_dir, file), 'r') as f:
                for group in f.keys():
                    self.len += f[group]['text'].shape[0]

        self.random_data = False

    def __len__(self):
        return self.len

    def load_file(self, idx):
        data = []
        file_path = self.files[idx]
        with h5py.File(os.path.join(self.data_dir, file_path), 'r') as f:
            for group in f.keys():
                length = f[group]['text'].shape[0]
                text = f[group]['text'][:]
                image = f[group]['image'][:]
                image_pooled = f[group]['image.pooled'][:]
                features = f[group]['features'][:]
                if self.random_data:
                    data.append({
                        "length": length,
                        "text": np.random.rand(*text.shape),
                        "image": np.random.rand(*image.shape),
                        "image.pooled": np.random.rand(*image_pooled.shape),
                        "features": np.random.rand(*features.shape),
                    })
                else: 
                    data.append({
                        # "subtask": group,
                        "length": f[group]['text'].shape[0],
                        "text": f[group]['text'][:],
                        "image": f[group]['image'][:],
                        "image.pooled": f[group]['image.pooled'][:],
                        "features": f[group]['features'][:],
                    })
        return data

    def __iter__(self):
        return self.iter_data()
    
    def iter_data(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # num_workers == 0
            file_iter_range = range(len(self.files))
        else:  
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            file_iter_range = range(worker_id * per_worker, min((worker_id + 1) * per_worker, len(self.files)))
        
        for idx in file_iter_range:
            current_data = self.load_file(idx)
            for group in current_data:
                for t in range(group["length"]):
                    text = torch.tensor(group["text"][t, 0, ...], dtype=torch.float32)  # (L, 2048)
                    image = torch.tensor(group["image"][t, 0, 0, 0, ...], dtype=torch.float32)  # (256+256, 1024)
                    image_pool = torch.tensor(group["image.pooled"][t, 0, ...], dtype=torch.float32)  # (768+768)
                    feature = torch.tensor(group["features"][t, 0, ...], dtype=torch.float32)  # (L, 2048)
                    if not self.random_data:
                        feature = (feature - self.norm_stats["feature_mean"]) / self.norm_stats["feature_std"]
                    # if t > 0:
                    #     prev_feature = torch.tensor(group["features"][t - 1, 0, ...], dtype=torch.float32) 
                    if t == 0:
                        prev_feature = torch.zeros_like(feature)
                    text_pool = torch.mean(text, dim=0)  # (2048,)
                    prev_feature_pool = torch.mean(prev_feature, dim=0)  # (2048,)
                    yield text, text_pool, image, image_pool, feature, prev_feature, prev_feature_pool
                    prev_feature = feature


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


def get_norm_stats(data_dir, data_size):
    files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    files = files[:data_size]
    # all_image_data = []
    # all_image_pooled_data = []
    # all_text_data = []
    all_feature_data = []
    for file in files:
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            for group in f.keys():
                # text = f[group]['text'][:].squeeze(1)  # (T, L, 2048)
                # image = f[group]['image'][:].squeeze(1).squeeze(1).squeeze(1)  # (T, 256+256, 1024)
                # image_pooled = f[group]['image.pooled'][:].squeeze(1)  # (T, 768+768)
                features = f[group]['features'][:].squeeze(1).reshape(-1, 2048)  # (T * L, 2048)
                # all_text_data.append(text)
                # all_image_data.append(image)
                # all_image_pooled_data.append(image_pooled)
                all_feature_data.append(torch.from_numpy(features))
    # all_text_data = torch.cat(all_text_data)
    # all_image_data = torch.cat(all_image_data)
    # all_image_pooled_data = torch.cat(all_image_pooled_data)
    all_feature_data = torch.cat(all_feature_data)

    # normalize feature data
    feature_mean = all_feature_data.mean(dim=0, keepdim=True)  # (1, 2048)
    feature_std = all_feature_data.std(dim=0, keepdim=True)  # (1, 2048)
    feature_std = torch.clip(feature_std, 1e-2, np.inf)  # clipping

    stats = {
        "feature_mean": feature_mean.numpy().squeeze(),
        "feature_std": feature_std.numpy().squeeze(),
    }

    return stats


def load_data(config, num_workers=8):
    data_dir = '/Share/xyli/Datasets/flamingo_data/logs_20241002_011950'
    stats = get_norm_stats(data_dir, config["data_size"])

    if config["iterable_data"]:
        train_dataset = MyIterableDataset(data_dir, stats, split='train', data_size=config["data_size"])
        val_dataset = MyIterableDataset(data_dir, stats, split='val', data_size=config["data_size"])
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=num_workers,
            prefetch_factor=config["batch_size"],
            pin_memory=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=config["batch_size"],
            pin_memory=False,
        )
    else:
        train_dataset = MyDataset(data_dir, stats, split='train', data_size=config["data_size"])
        val_dataset = MyDataset(data_dir, stats, split='val', data_size=config["data_size"])
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn, 
            num_workers=num_workers,
            prefetch_factor=config["batch_size"],
            pin_memory=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=config["batch_size"],
            pin_memory=False,
        )

    return train_dataloader, val_dataloader, stats