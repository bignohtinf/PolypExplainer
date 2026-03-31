import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor

from src.config import Config
from src.models.llm_wrapper import OllamaExplainer
from src.models.segformer import get_segformer_model
from src.xai.morphology import extract_polyp_features


def run_explanation(image_path: str, model_path: str):
    device = torch.device(Config.DEVICE)
    processor = SegformerImageProcessor.from_pretrained(Config.MODEL_NAME)
    model = get_segformer_model(Config.MODEL_NAME, num_classes=Config.NUM_CLASSES)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    input_tensor = inputs["pixel_values"]

    os.makedirs("outputs/predictions", exist_ok=True)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor.to(device))
        logits = outputs.logits
        h, w = image.size[1], image.size[0]
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        pred_mask = (torch.softmax(upsampled_logits, dim=1)[0, 1] > 0.5).float().cpu().numpy()

    features = extract_polyp_features(pred_mask)
    explainer = OllamaExplainer(
        model_name=Config.OLLAMA_MODEL,
        base_url=Config.OLLAMA_API_URL,
    )
    report = explainer.generate_explanation(features)

    print("-" * 30)
    print("BÁO CÁO TỪ AI:")
    print(report)
    print("-" * 30)

    cv2.imwrite("outputs/predictions/result.png", (pred_mask * 255).astype(np.uint8))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Giải thích kết quả phân đoạn polyp (SegFormer + Ollama).")
    parser.add_argument("--image", required=True, help="Đường dẫn file ảnh đầu vào")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(Config.CHECKPOINT_DIR, "segformer_polyp_best.pth"),
        help="File checkpoint .pth",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"Không tìm thấy ảnh: {args.image}")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"Không tìm thấy checkpoint: {args.checkpoint}")

    run_explanation(args.image, args.checkpoint)
