import streamlit as st
import numpy as np
import torch
import clip
from PIL import Image
import os

checkpoint_path = r"Model/clip_fashion_export.pt"
folder_path = r"archive/fashion-dataset/images"
features_path = "image_features.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# 1) Load CLIP model
# ======================================================
@st.cache_resource
def load_clip_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]

    model, preprocess = clip.load(model_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    return model, preprocess


model, preprocess = load_clip_model(checkpoint_path, device)


# ======================================================
# 2) Tính feature và lưu lại vào .pt (chạy 1 lần)
# ======================================================
def compute_and_save_features(folder_path, max_images=2000):
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:max_images]

    images_tensor = []
    valid_files = []

    for f in files:
        try:
            img = Image.open(os.path.join(folder_path, f)).convert("RGB")
            img_tensor = preprocess(img).to(device)
            images_tensor.append(img_tensor)
            valid_files.append(f)
        except:
            continue

    images_tensor = torch.stack(images_tensor).to(device)

    with torch.no_grad():
        features = model.encode_image(images_tensor).float().cpu()

    # 🔥 Lưu dưới dạng dictionary chuẩn
    torch.save({
        "image_files": valid_files,
        "image_features": features
    }, features_path)

    return valid_files, features


# ======================================================
# 3) Load feature từ file nếu có, nếu không thì tính
# ======================================================
@st.cache_resource
def load_saved_features():
    if os.path.exists(features_path):
        data = torch.load(features_path, map_location="cpu")
        return data["image_files"], data["image_features"]
    else:
        return compute_and_save_features(folder_path, max_images=2000)


image_files, image_features = load_saved_features()
image_features = image_features.to(device)


# ======================================================
# 4) UI tìm kiếm
# ======================================================
title = st.title("🔍 Text to Image Search with CLIP")
content = st.markdown(
    """
    This app allows you to search for images using text queries.
    It uses a pre-trained CLIP model to compute image and text features.
    """
)
query = st.text_input("Enter your search query:")

if query:
    tokens = clip.tokenize(["This is " + query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens).float()

    img_f = image_features / image_features.norm(dim=-1, keepdim=True)
    txt_f = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = (txt_f @ img_f.T).squeeze().cpu().numpy()

    top_k = st.slider("Top K", 1, 30, 5)
    idxs = np.argsort(-similarity)[:top_k]

    cols = st.columns(3)  # mỗi hàng 3 ảnh

    for idx, i in enumerate(idxs):
        col = cols[idx % 3]  # chia đều vào 3 cột

        with col:
            img_path = os.path.join(folder_path, image_files[i])
            caption = f"{image_files[i]} - Similarity: {similarity[i]:.4f}"

            st.image(img_path, caption=caption, use_container_width=True)
        
        # Xuống hàng sau mỗi 3 ảnh
        if (idx + 1) % 3 == 0:
            cols = st.columns(3)

