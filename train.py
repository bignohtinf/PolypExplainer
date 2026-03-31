import argparse
import json
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.dataset import KvasirSegDataset, get_transforms
from src.models.segformer import get_segformer_model
from src.utils import (
    dice_score,
    iou_score,
    load_checkpoint,
    load_training_history,
    save_checkpoint,
    save_full_checkpoint,
    save_training_history,
)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        dice = dice_score(upsampled_logits, labels)
        total_loss += loss.item()
        total_dice += dice.item()

        pbar.set_postfix({"Loss": loss.item(), "Dice": dice.item()})

    return total_loss / len(loader), total_dice / len(loader)


def validate(model, loader, device):
    model.eval()
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )

            total_dice += dice_score(upsampled_logits, labels).item()
            total_iou += iou_score(upsampled_logits, labels).item()

    return total_dice / len(loader), total_iou / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện SegFormer polyp segmentation.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục từ output/checkpoints/checkpoint.pth (model + optimizer + epoch + best_dice).",
    )
    args = parser.parse_args()

    device = torch.device(Config.DEVICE)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    with open(Config.SPLIT_JSON, "r") as f:
        splits = json.load(f)

    train_tf, val_tf = get_transforms(img_size=Config.IMG_SIZE)

    train_ds = KvasirSegDataset(splits["train"], Config.IMAGE_DIR, Config.MASK_DIR, transform=train_tf)
    val_ds = KvasirSegDataset(splits["val"], Config.IMAGE_DIR, Config.MASK_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = get_segformer_model(Config.MODEL_NAME, num_classes=Config.NUM_CLASSES)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    best_dice = 0.0
    start_epoch = 0
    history = []

    if args.resume:
        last_completed, best_dice = load_checkpoint(
            model,
            optimizer,
            Config.CHECKPOINT_DIR,
            scheduler=None,
            map_location=device,
        )
        if last_completed >= 0:
            start_epoch = last_completed + 1
        else:
            start_epoch = 0
            if os.path.exists(os.path.join(Config.CHECKPOINT_DIR, "net_latest.pth")):
                print("Tiếp tục chỉ với weights (net_latest); optimizer khởi tạo lại, epoch bắt đầu từ 1.")
        history = load_training_history(Config.CHECKPOINT_DIR)

    if start_epoch >= Config.EPOCHS:
        print(f"Đã huấn luyện đủ {Config.EPOCHS} epoch (start_epoch={start_epoch}). Không chạy thêm.")
        return

    for epoch in range(start_epoch, Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, device)
        val_dice, val_iou = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(Config.CHECKPOINT_DIR, "segformer_polyp_best.pth")
            save_checkpoint(model, best_path)
            print(f"--- Đã lưu model tốt nhất với Dice: {best_dice:.4f} ---")

        save_full_checkpoint(
            model,
            optimizer,
            Config.CHECKPOINT_DIR,
            epoch=epoch,
            best_dice=best_dice,
            scheduler=None,
        )
        print(f"Đã lưu checkpoint resume (epoch đã xong: {epoch + 1}/{Config.EPOCHS}).")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_dice": round(train_dice, 6),
                "val_dice": round(val_dice, 6),
                "val_iou": round(val_iou, 6),
                "best_dice": round(best_dice, 6),
            }
        )
        save_training_history(Config.CHECKPOINT_DIR, history)


if __name__ == "__main__":
    main()
