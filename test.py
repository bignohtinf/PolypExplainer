import torch
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import KvasirSegDataset, get_transforms
from src.models.segformer import get_segformer_model
from src.utils import dice_score, iou_score

def run_test():
    device = torch.device(Config.DEVICE)
    
    # 1. Load Split
    with open(Config.SPLIT_JSON, 'r') as f:
        splits = json.load(f)
    
    # 2. Prepare DataLoader
    _, val_tf = get_transforms(img_size=Config.IMG_SIZE)
    test_ds = KvasirSegDataset(splits["test"], Config.IMAGE_DIR, Config.MASK_DIR, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Load Model
    model = get_segformer_model(Config.MODEL_NAME, num_classes=Config.NUM_CLASSES)
    ckpt = os.path.join(Config.CHECKPOINT_DIR, "segformer_polyp_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    # 4. Evaluation
    total_dice, total_iou = 0, 0
    os.makedirs("outputs/predictions", exist_ok=True)

    print("Đang đánh giá trên tập Test...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            file_id = batch["id"][0]

            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            
            # Upscale
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            
            # Tính metric
            total_dice += dice_score(upsampled_logits, labels).item()
            total_iou += iou_score(upsampled_logits, labels).item()

            # Lưu thử 5 ảnh đầu tiên để xem mắt thường
            if i < 5:
                probs = torch.softmax(upsampled_logits, dim=1)[0, 1].cpu().numpy()
                pred_mask = (probs > 0.5).astype("uint8")
                cv2.imwrite(f"outputs/predictions/{file_id}_pred.png", pred_mask * 255)

    print(f"\n--- KẾT QUẢ TẬP TEST ---")
    print(f"Mean Dice: {total_dice / len(test_loader):.4f}")
    print(f"Mean IoU: {total_iou / len(test_loader):.4f}")

if __name__ == "__main__":
    run_test()