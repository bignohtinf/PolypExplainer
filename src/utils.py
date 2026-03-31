import torch
import numpy as np
import json
import os
import glob
import re

def dice_score(preds, targets, threshold=0.5):
    if preds.shape[1] == 1:
        preds = (torch.sigmoid(preds) > threshold).float()
    else:
        probs = torch.softmax(preds, dim=1)[:, 1]
        preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    return (2.0 * intersection) / (preds.sum() + targets.sum() + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    if preds.shape[1] == 1:
        preds = (torch.sigmoid(preds) > threshold).float()
    else:
        probs = torch.softmax(preds, dim=1)[:, 1]
        preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def save_full_checkpoint(
    model,
    optimizer,
    save_dir: str,
    epoch: int,
    best_dice: float,
    scheduler=None,
):
    """Lưu trạng thái đầy đủ để resume (epoch = chỉ số epoch vừa hoàn thành, 0-based)."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dice": float(best_dice),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    path = os.path.join(save_dir, "checkpoint.pth")
    torch.save(payload, path)
    save_checkpoint(model, os.path.join(save_dir, "net_latest.pth"))
    
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'net_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            if result:
                epochs_exist.append(int(result[0]))
        if epochs_exist:
            initial_epoch = max(epochs_exist)
        else:
            initial_epoch = 0
    else:
        initial_epoch = 0

    return initial_epoch

def load_checkpoint(model, optimizer, save_dir: str, scheduler=None, map_location=None):
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    latest_path = os.path.join(save_dir, "net_latest.pth")

    if os.path.exists(checkpoint_path):
        print(f"Đang tải checkpoint đầy đủ: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        last_completed_epoch = int(checkpoint["epoch"])
        best_dice = float(checkpoint.get("best_dice", checkpoint.get("best_psnr", 0.0)))
        print(
            f"Resume: đã hoàn thành tới epoch {last_completed_epoch + 1} (0-based last={last_completed_epoch}), "
            f"best_dice={best_dice:.4f}"
        )
        return last_completed_epoch, best_dice

    if os.path.exists(latest_path):
        print(f"Chỉ có weights mới nhất (không có optimizer): {latest_path}")
        model.load_state_dict(torch.load(latest_path, map_location=map_location))
        # Không có optimizer state: train lại từ epoch 0 với weights hiện có
        return -1, 0.0

    return -1, 0.0

def load_training_history(save_dir):
    """Load training history from JSON file"""
    history_path = os.path.join(save_dir, 'training_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f"Loaded training history with {len(history)} epochs")
            return history
        except Exception as e:
            print(f"Warning: Could not load training history: {e}")
            return []
    return []

def cleanup_old_models(save_dir, keep_latest=True, keep_best=True):
    """Delete old epoch models, keeping only latest and best"""
    file_list = glob.glob(os.path.join(save_dir, 'net_epoch*.pth'))
    
    if not file_list:
        return
    
    # Keep latest and best
    files_to_keep = set()
    if keep_latest:
        latest_path = os.path.join(save_dir, 'net_latest.pth')
        if os.path.exists(latest_path):
            files_to_keep.add(latest_path)
    
    if keep_best:
        best_path = os.path.join(save_dir, 'net_best.pth')
        if os.path.exists(best_path):
            files_to_keep.add(best_path)
    
    # Delete old epoch models
    deleted_count = 0
    for file_path in file_list:
        if file_path not in files_to_keep:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old model file(s)")

def save_training_history(save_dir, history):
    """Save training history to JSON file"""
    history_path = os.path.join(save_dir, 'training_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save training history: {e}")