import cv2
import numpy as np

def extract_polyp_features(mask):
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "Không phát hiện polyp"
    features = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter**2 + 1e-8)
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        desc = {
            "id": i + 1,
            "area_pixel": area,
            "shape": "perfectly round" if circularity > 0.8 else "abnormal/long",
            "position": f"center coordinates ({x + w // 2}, {y + h //2})",
            "size_relative": "big" if area > 5000 else "small"
        }
        
        features.append(desc)
        
    return features
