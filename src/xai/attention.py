import torch 
import torch.nn.functional as F 
import numpy as np
import cv2

def extract_attention_map(outputs, target_size=(512, 512)):
    attentions = outputs.attentions
    last_layer_attn = attentions[-1]
    
    avg_attn = torch.mean(last_layer_attn, dim=1)[0]
    return avg_attn

def get_heatmap(mask, image_size=(512, 512)):
    mask = (mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return heatmap
