"""
Training Pipeline for MultimodalVideoProctorNet.
Implements AMP FP16, AdamW, CosineAnnealingWarmRestarts, Gradient Accumulation,
Multiclass Focal Loss, and Validation Macro F1 checkpointing.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from video_model import get_model
from losses import MulticlassFocalLoss
from dataset_loader import get_dataloaders


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    accum_steps: int = 2,
    use_amp: bool = True
) -> float:
    """Trains the model for one epoch with AMP and gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for step, (videos, audios, targets) in enumerate(dataloader):
        videos = videos.to(device, non_blocking=True)
        audios = audios.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp and device.type == 'cuda'):
            logits = model(videos, audios)
            loss = criterion(logits, targets)
            loss = loss / accum_steps

        # Backward pass with GradScaler
        if use_amp and device.type == 'cuda':
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> dict:
    """Evaluates validation loss, Macro F1, Precision, and Recall."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for videos, audios, targets in dataloader:
        videos = videos.to(device, non_blocking=True)
        audios = audios.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp and device.type == 'cuda'):
            logits = model(videos, audios)
            loss = criterion(logits, targets)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    acc = np.mean(all_preds == all_targets)

    return {
        'loss': avg_loss,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'accuracy': acc
    }


def main():
    parser = argparse.ArgumentParser(description="Train MultimodalVideoProctorNet")
    parser.add_argument("--dataset", type=str, default="ml/data/proctoring_dataset.h5")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="ml/models/best_proctor_model.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting MultimodalVideoProctorNet Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        hdf5_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    print(f"Dataset partitions loaded: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val")

    # Initialize 36-Layer, ~45M Model
    model = get_model(num_classes=4).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated: 36 Layers, {num_params:,} parameters (~{num_params / 1e6:.1f}M)")

    # Loss Function: Multiclass Focal Loss with class weighting
    class_weights = torch.tensor([0.25, 0.75, 0.75, 0.75], device=device)
    criterion = MulticlassFocalLoss(alpha=class_weights, gamma=2.0)

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )

    # Scheduler: CosineAnnealingWarmRestarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_f1 = 0.0
    patience_counter = 0

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accum_steps=args.accum_steps,
            use_amp=(device.type == 'cuda')
        )

        val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=(device.type == 'cuda')
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        curr_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2%} | "
            f"LR: {curr_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save best checkpoint
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val_f1,
                'val_metrics': val_metrics,
                'num_params': num_params
            }, args.save_path)
            print(f"  >>> Checkpoint saved to {args.save_path} (New Best Val Macro-F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs without improvement.")
                break

    total_duration = time.time() - total_start_time
    print(f"=== Training Completed in {total_duration / 60:.1f} minutes ===")
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
