import json
import os

from sklearn.model_selection import train_test_split

from src.config import Config


def main():
    os.makedirs(os.path.dirname(Config.SPLIT_JSON), exist_ok=True)

    image_dir = Config.IMAGE_DIR
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Thư mục ảnh không tồn tại: {image_dir}. Cập nhật Config.IMAGE_DIR cho đúng dataset."
        )

    all_ids = [
        f.split(".")[0]
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    all_ids = sorted(set(all_ids))
    if len(all_ids) < 3:
        raise RuntimeError(f"Cần ít nhất 3 ảnh để chia split, hiện có {len(all_ids)} trong {image_dir}")

    train_ids, tmp_ids = train_test_split(all_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(tmp_ids, test_size=0.5, random_state=42)

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(Config.SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Đã ghi {Config.SPLIT_JSON}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")


if __name__ == "__main__":
    main()
