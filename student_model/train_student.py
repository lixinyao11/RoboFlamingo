import torch
import os
import numpy as np
import sys
from torch import nn, optim
import wandb
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("/home/xyli/Code/RoboFlamingo")
from student_model.data import load_data
from student_model.model import GPT2FeaturePrediction
from student_model.distributed import init_distributed_device


def forward_pass(model, batch, criterion, device):
    text, image, image_pool, target_feature, prev_feature, mask = [t.to(device) for t in batch]
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
    device_id, rank = init_distributed_device()
    print("device_id: ", device_id)

    train_dataloader, val_dataloader = load_data(config)

    learning_rate = config["lr"]
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]

    # Instantiate the model, loss function, and optimizer
    model = GPT2FeaturePrediction(config).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
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
    optimizer = optim.AdamW(get_grouped_params(ddp_model), lr=learning_rate)

    # if args.lr_scheduler == "linear":
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=args.warmup_steps,
    #         num_training_steps=total_training_steps,
    #     )
    # elif args.lr_scheduler == "cosine":
    #     lr_scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=args.warmup_steps,
    #         num_training_steps=total_training_steps,
    #     )
    # elif args.lr_scheduler == 'cosine_restart':
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    # else:
    #     lr_scheduler = get_constant_schedule_with_warmup(
    #         optimizer, num_warmup_steps=args.warmup_steps
    #     )

    if rank == 0:
        wandb.init(
            project="student_flamingo",
            name=config["exptid"],
            entity="xinyaoli-sjtu-icec",
            dir="/Share/xyli/logs/flamingo/logs",
        )
        wandb.config.update(config)

    ddp_model.train()
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")

        if epoch % 5 == 0 and epoch > 0 and rank == 0:
            # val_dataloader = get_data_loader(val_dataset, config)
            with torch.inference_mode():
                ddp_model.eval()
                val_loss = 0.0
                for i, batch in enumerate(val_dataloader):
                    loss = forward_pass(ddp_model, batch, criterion, device_id)
                    val_loss += loss.item()
                    if i % 100 == 0 and rank == 0:
                        print(f"Val batch {i}, loss: {loss.item():.4f}")
                val_loss /= len(val_dataloader)
            wandb.log({"val/loss": val_loss})
            print(f"Val loss: {val_loss:.5f}")

        ddp_model.train()
        running_loss = 0.0
        # train_dataloader = get_data_loader(train_dataset, config)
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = forward_pass(ddp_model, batch, criterion, device_id)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0 and rank == 0:
                wandb.log({"loss": loss.item()})
                print(f"Train batch {i}, loss: {loss.item():.4f}")

        if rank == 0:
            avg_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            os.makedirs(os.path.join(ckpt_dir, config["exptid"]), exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, config["exptid"], f"model_epoch_{epoch}.ckpt")
            torch.save(get_checkpoint(ddp_model), ckpt_path)

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
        "batch_size": 6,
        "exptid": "test_run_gpt_ddp",
        "data_size": 100,
        "iterable_data": True,
        "weight_decay": 0.01,
    }
    main(config)
