import os
from typing import Any

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class KvasirSegDataset(Dataset):
    """Dataset built from file IDs (no extension) and root image/mask directories."""

    def __init__(
        self,
        file_ids: list[str],
        image_dir: str,
        mask_dir: str,
        bbox_data: dict[str, Any] | None = None,
        transform=None,
    ):
        self.file_ids = list(file_ids)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_data = bbox_data
        self.transform = transform

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx: int):
        file_id = self.file_ids[idx]
        image_path = os.path.join(self.image_dir, f"{file_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{file_id}.jpg")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Không đọc được mask: {mask_path}")

        mask = (mask > 127).astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        meta = self.bbox_data.get(file_id, {}) if self.bbox_data else {}

        return {
            "pixel_values": image,
            "labels": torch.as_tensor(mask, dtype=torch.long),
            "id": file_id,
            "bbox_info": str(meta),
        }


def get_transforms(img_size=512):
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform
