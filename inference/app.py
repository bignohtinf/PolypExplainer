import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os
from src.config import Config
from inference import inference
from src.xai.morphology import extract_polyp_features
from src.models.llm_wrapper import OllamaExplainer

# --- Cấu hình trang ---
st.set_page_config(page_title="Polyp Diagnosis AI", layout="wide")
st.title("🩺 Hệ thống Chẩn đoán Polyp Đại tràng tích hợp Giải thích (XAI)")

# --- Sidebar ---
with st.sidebar:
    st.header("Cấu hình")
    model_path = st.text_input(
        "Đường dẫn Model (.pth)", 
        os.path.join(Config.CHECKPOINT_DIR, "segformer_polyp_final.pth")
    )
    threshold = st.slider("Ngưỡng tin cậy (Threshold)", 0.1, 0.9, 0.5)
    st.info("Đảm bảo Ollama đang chạy ở Local!")

# --- Main UI ---
uploaded_file = st.file_uploader("Tải ảnh nội soi lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Hiển thị ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)

    # 2. Chạy Inference khi nhấn nút
    if st.button("Phân tích Polyp"):
        with st.spinner("Đang phân tích và sinh lời giải thích..."):
            # Lưu ảnh tạm
            temp_path = "temp_input.jpg"
            image.save(temp_path)
            
            try:
                # Chạy SegFormer
                device = torch.device(Config.DEVICE)
                pred_mask, confidence, image_raw = inference(temp_path, model_path, device)
                
                # Tạo ảnh Overlay
                mask_uint8 = (pred_mask * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(image_raw, 0.7, heatmap, 0.3, 0)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                with col2:
                    st.subheader("Kết quả phân vùng")
                    st.image(overlay_rgb, use_container_width=True)

                # 3. Trích xuất đặc trưng & Gọi LLM
                polyp_features = extract_polyp_features(pred_mask)
                
                # Tạo dict chứa tất cả thông tin
                features = {
                    "polyps": polyp_features,
                    "mean_confidence": float(confidence.mean()),
                    "max_confidence": float(confidence.max())
                }
                
                st.divider()
                
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.subheader("📊 Thông số kỹ thuật")
                    st.json(features)

                with col_right:
                    st.subheader("📝 Báo cáo giải thích từ AI")
                    explainer = OllamaExplainer()
                    report = explainer.generate_explanation(features)
                    st.write(report)
                    
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
            finally:
                # Xóa ảnh tạm
                if os.path.exists(temp_path):
                    os.remove(temp_path)