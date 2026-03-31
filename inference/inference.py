import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from src.config import Config
from src.models.segformer import get_segformer_model
import torchvision.transforms as T


def load_test_ids_from_split(split_json_path):
    if not os.path.exists(split_json_path):
        raise FileNotFoundError(f"Split file not found: {split_json_path}")
    
    with open(split_json_path, 'r') as f:
        splits = json.load(f)
    
    test_ids = splits.get("val", [])
    return test_ids


def inference(image_path, model_path, device):
    model = get_segformer_model(Config.MODEL_NAME, num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    image_raw = cv2.imread(image_path)
    if image_raw is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    h_orig, w_orig = image_raw.shape[:2]
    image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(h_orig, w_orig),
            mode='bilinear',
            align_corners=False
        )
        
        probs = torch.sigmoid(upsampled_logits)
        pred_mask = (probs > 0.5).float().cpu().numpy()[0][0]
        confidence = probs.cpu().numpy()[0][0]
    
    return pred_mask, confidence, image_raw


def save_prediction(mask, confidence, output_dir, file_id):
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_path = os.path.join(output_dir, f"{file_id}_mask.png")
    cv2.imwrite(mask_path, mask_uint8)
    return mask_path


def main():
    output_dir = "output/predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(Config.DEVICE)
    model_path = os.path.join(Config.CHECKPOINT_DIR, "segformer_polyp_final.pth")
    
    # Load test IDs
    test_ids = load_test_ids_from_split(Config.SPLIT_JSON)
    
    # Random 10 images
    np.random.seed(42)
    sample_ids = np.random.choice(test_ids, size=min(10, len(test_ids)), replace=False)
    
    print(f"Running inference on {len(sample_ids)} random test images...\n")
    
    for file_id in tqdm(sample_ids):
        try:
            image_path = os.path.join(Config.IMAGE_DIR, f"{file_id}.jpg")
            pred_mask, confidence, _ = inference(image_path, model_path, device)
            save_prediction(pred_mask, confidence, output_dir, file_id)
        except Exception as e:
            print(f"Error: {file_id} - {str(e)}")
    
    print(f"\nPredictions saved to {output_dir}")


if __name__ == "__main__":
    main()
    