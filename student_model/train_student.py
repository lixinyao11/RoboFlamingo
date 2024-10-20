import torch
import os
import numpy as np
import pickle
import sys
from torch import nn, optim
import wandb
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("/home/xyli/Code/RoboFlamingo")
from student_model.data import load_data
from student_model.model import GPT2FeaturePrediction
from student_model.distributed import init_distributed_device

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def forward_pass(model, batch, criterion, device, cast_dtype=torch.float32):
    text, image, image_pool, target_feature, prev_feature, mask = [t.to(device, dtype=cast_dtype) for t in batch]
    predicted_feature = model(text, image, image_pool, prev_feature, mask)
    L = target_feature.shape[1]
    predicted_feature = predicted_feature[..., -L:, :]  # (B, L, 2048)
    assert predicted_feature.shape == target_feature.shape, (predicted_feature.shape, target_feature.shape)
    loss = criterion(predicted_feature, target_feature)  # (B, L, 2048)
    return loss


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    return state_dict


def main(config):
    train_dataloader, val_dataloader, stats = load_data(config)

    ckpt_dir = config["ckpt_dir"]
    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    data_size = config["data_size"]
    stats_path = os.path.join(ckpt_dir, f"dataset_{data_size}_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    num_epochs = config["num_epochs"]

    model = GPT2FeaturePrediction(config)
    if config["precision"] == "bf16":
        cast_dtype = torch.bfloat16
        model = model.bfloat16()
    elif config["precision"] == "fp16":
        cast_dtype = torch.float16
        model = model.half()
    else:
        cast_dtype = torch.float32
        model = model.float()

    if config["device"] == "cpu":
        device_id = "cpu"
        model = model.cpu()
    else:
        device_id, rank = init_distributed_device()
        print("device_id: ", device_id)
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": config["weight_decay"]},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(get_grouped_params(model), lr=config["lr"])
    total_training_steps = len(train_dataloader) * num_epochs // config["batch_size"]
    if config["lr_scheduler"] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * total_training_steps,
            num_training_steps=total_training_steps,
        )
    elif config["lr_scheduler"] == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * total_training_steps,
            num_training_steps=total_training_steps,
        )
    elif config["lr_scheduler"] == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * total_training_steps
        )

    if config["device"] =="cpu" or rank == 0:
        wandb.init(
            project="student_flamingo",
            name=config["exptid"],
            entity="xinyaoli-sjtu-icec",
            dir="/Share/xyli/logs/flamingo/logs",
        )
        wandb.config.update(config)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")

        if epoch % 5 == 0 and epoch > 0 and rank == 0:
            # val_dataloader = get_data_loader(val_dataset, config)
            with torch.inference_mode():
                model.eval()
                val_loss = 0.0
                for i, batch in enumerate(val_dataloader):
                    loss = forward_pass(model, batch, criterion, device_id, cast_dtype)
                    val_loss += loss.item()
                    if i % 100 == 0 and rank == 0:
                        print(f"Val batch {i}, loss: {loss.item():.4f}")
                val_loss /= len(val_dataloader)
            wandb.log({"val/loss": val_loss, "epoch": epoch})
            print(f"Val loss: {val_loss:.5f}")

        model.train()
        running_loss = 0.0
        # train_dataloader = get_data_loader(train_dataset, config)
        for i, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            loss = forward_pass(model, batch, criterion, device_id, cast_dtype)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            if i % 100 == 0 and (device_id == 'cpu' or rank == 0):
                # wandb.log({"loss": loss.item()})
                print(f"Train batch {i}, loss: {loss.item():.4f}")

        if device_id == 'cpu' or rank == 0:
            avg_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            wandb.log({"train/loss": avg_loss, "epoch": epoch})
            os.makedirs(os.path.join(ckpt_dir, config["exptid"]), exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, config["exptid"], f"model_epoch_{epoch}.ckpt")
            torch.save(get_checkpoint(model), ckpt_path)

    # wandb.finish()


if __name__ == "__main__":
    config = {
        "num_epochs": 80,
        "ckpt_dir": "/Share/xyli/logs/flamingo/ckpts",
        "vision_dim": 1024,
        "vision_pool_dim": 1536,
        "embed_dim": 2048,
        "feat_dim": 2048,
        "lr": 1e-4,
        "lr_scheduler": "constant",
        "batch_size": 16,  # 4
        "precision": "float32",
        "exptid": "gpt_D",
        "data_size": 100,  # 20
        "iterable_data": False,
        "weight_decay": 0.01,
        "device": "cuda",
    }
    main(config)
