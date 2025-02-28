import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from block_model import BlockAutoencoder
from dataset import BlockDataset
from tqdm import tqdm
import os
import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def create_loaders(dataset, batch_size=64, val_ratio=0.1):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader

def compute_loss(model, outputs, batch):
    block_id, numeric_features, segment = batch
    grid_x = numeric_features[:, 0].long()   # grid_x
    grid_y = numeric_features[:, 1].long()   # grid_y
    angle_target = numeric_features[:, 2:4]  # sin, cos – целевые значения
    flip_x = numeric_features[:, 4]
    flip_y = numeric_features[:, 5]
    
    id_emb_target = model.id_embed(block_id)
    pos_x_emb_target = model.pos_x_embed(grid_x)
    pos_y_emb_target = model.pos_y_embed(grid_y)
    seg_emb_target = model.seg_embed(segment)
    
    # меняем точность признаков
    losses = {
        "id": 1.5*F.mse_loss(outputs["id_emb"], id_emb_target),
        "pos_x": 1.2*F.mse_loss(outputs["pos_x_emb"], pos_x_emb_target),
        "pos_y": 1.2*F.mse_loss(outputs["pos_y_emb"], pos_y_emb_target),
        "angle": 0.7*F.mse_loss(outputs["angle_sin_cos"], angle_target),
        "flip_x": 1.1 * F.binary_cross_entropy_with_logits(outputs["flip_x"].squeeze(), flip_x),
        "flip_y": 1.1 * F.binary_cross_entropy_with_logits(outputs["flip_y"].squeeze(), flip_y),
        "segment": 1.2*F.mse_loss(outputs["segment_emb"], seg_emb_target)
    }
    total_loss = sum(losses.values())
    return total_loss, losses

def validate_indices(model, batch):
    block_id, numeric, segment = batch

    assert (block_id < model.id_embed.num_embeddings).all(), \
        f"ID index out of range: max {model.id_embed.num_embeddings-1}"
    grid_x = numeric[:, 0].long()
    grid_y = numeric[:, 1].long()
    assert (grid_x < model.pos_x_embed.num_embeddings).all(), \
        f"X index out of range: max {model.pos_x_embed.num_embeddings-1}"
    assert (grid_y < model.pos_y_embed.num_embeddings).all(), \
        f"Y index out of range: max {model.pos_y_embed.num_embeddings-1}"
    assert (segment < model.seg_embed.num_embeddings).all(), \
        f"Segment index out of range: max {model.seg_embed.num_embeddings-1}"

def train_epoch(model, loader, optimizer, device):
    """Обработка одной эпохи тренировки"""
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc="Training", leave=False)
    
    for batch in progress:
        validate_indices(model, batch)
        block_id = batch[0].to(device)
        numeric = batch[1].to(device)
        segment = batch[2].to(device)
        
        # numeric: [grid_x, grid_y, sin, cos, flip_x, flip_y]
        grid_x = numeric[:, 0].long()
        grid_y = numeric[:, 1].long()
        pos_x = grid_x * model.grid_size
        pos_y = grid_y * model.grid_size
        sin_val = numeric[:, 2]
        cos_val = numeric[:, 3]
        flip_x_val = numeric[:, 4]
        flip_y_val = numeric[:, 5]
        
        # вход: (block_id, pos_x, pos_y, sin, cos, flip_x, flip_y, segment)
        inputs = (block_id, pos_x, pos_y, sin_val, cos_val, flip_x_val, flip_y_val, segment)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, loss_components = compute_loss(model, outputs, (block_id, numeric, segment))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})
    
    return total_loss / len(loader)

def validate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            validate_indices(model, batch)
            
            block_id = batch[0].to(device)
            numeric = batch[1].to(device)
            segment = batch[2].to(device)
            
            grid_x = numeric[:, 0].long()
            grid_y = numeric[:, 1].long()
            pos_x = grid_x * model.grid_size
            pos_y = grid_y * model.grid_size
            sin_val = numeric[:, 2]
            cos_val = numeric[:, 3]
            flip_x_val = numeric[:, 4]
            flip_y_val = numeric[:, 5]
            
            inputs = (block_id, pos_x, pos_y, sin_val, cos_val, flip_x_val, flip_y_val, segment)
            outputs = model(inputs)
            loss, _ = compute_loss(model, outputs, (block_id, numeric, segment))
            total_loss += loss.item()
    
    return total_loss / len(loader)

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, train_loss, val_loss, dataset, config):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "id_to_idx": dataset.id_to_idx,
        "idx_to_id": dataset.idx_to_id,
        "max_x_idx": dataset.get_max_x_idx(),
        "max_y_idx": dataset.get_max_y_idx(),
        "grid_resolution": dataset.grid_resolution
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def train_model(data_dir, num_epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-5, resume_checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BlockDataset(data_dir)
    max_coord = max(dataset.get_max_x_idx(), dataset.get_max_y_idx()) * dataset.grid_resolution

    config = {
        "num_ids": dataset.get_num_ids(),
        "grid_size": dataset.grid_resolution,
        "max_coord": max_coord,
        "num_segments": dataset.get_max_seg_idx() + 1,
        "id_embed_dim": 10,
        "pos_embed_dim": 10,
        "seg_embed_dim": 4,
        "hidden_dim": 128,
        "latent_dim": 16
    }
    
    model = BlockAutoencoder(
        num_ids=config["num_ids"],
        grid_size=config["grid_size"],
        max_coord=config["max_coord"],
        num_segments=config["num_segments"],
        id_embed_dim=config["id_embed_dim"],
        pos_embed_dim=config["pos_embed_dim"],
        seg_embed_dim=config["seg_embed_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    
    start_epoch = 1
    best_val_loss = float("inf")

    if resume_checkpoint_path is not None and os.path.isfile(resume_checkpoint_path):
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        # Если в чекпоинте сохранена конфигурация датасета, её можно использовать при необходимости
        dataset.id_to_idx = checkpoint.get("id_to_idx", dataset.id_to_idx)
        dataset.idx_to_id = checkpoint.get("idx_to_id", dataset.idx_to_id)
        print(f"Resumed training from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
    else:
        print("Starting training from scratch.")

    print("Model parameters:")
    print(f"  - ID embeddings: {model.id_embed.num_embeddings}")
    print(f"  - Position embeddings (X/Y): {model.pos_x_embed.num_embeddings}")
    print(f"  - Segment embeddings: {model.seg_embed.num_embeddings}")
    print("Dataset stats:")
    print(f"  - Max grid X: {dataset.get_max_x_idx()}")
    print(f"  - Max grid Y: {dataset.get_max_y_idx()}")
    print(f"  - Max segment: {dataset.get_max_seg_idx()}")

    train_loader, val_loader = create_loaders(dataset, batch_size)

    for epoch in range(start_epoch, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_model(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f"models_16/epoch_{epoch}/model.pth"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                            epoch, train_loss, val_loss, dataset, config)
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    train_model(
        data_dir="datasets/pickles1",
        num_epochs=100,
        batch_size=512,
        lr=1e-4,
        resume_checkpoint_path=None
    )
