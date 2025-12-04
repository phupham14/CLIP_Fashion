import streamlit as st
import numpy as np
import torch
import clip
from PIL import Image
import os
checkpoint_path = r"Model\\clip_fashion_export.pt"
folder_path = r"archive\\fashion-dataset\\images"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint['model_name']
    model, preprocess = clip.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, preprocess

model, preprocess = load_clip_model(checkpoint_path, device)

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

# st.write("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# st.write("Input resolution:", input_resolution)
# st.write("Context length:", context_length)
# st.write("Vocab size:", vocab_size)

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)    
print("Context length:", context_length)
print("Vocab size:", vocab_size)

@st.cache_data
def get_image_features(folder_path, device):
    model, preprocess = load_clip_model(checkpoint_path, device)

    original_images, images = [], []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    st.write(f"Found {len(image_files)} image files.")

    valid_image_files = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Skip corrupted images
        try:
            image = Image.open(image_path)
            image.load()               # detect corrupted images
            image = image.convert("RGB")
        except Exception as e:
            print(f"⚠️ Lỗi đọc ảnh {image_file}: {e}")
            continue

        original_images.append(image)
        preprocessed_image = preprocess(image).to(device)
        images.append(preprocessed_image)
        valid_image_files.append(image_file)

    if len(images) == 0:
        st.error("Không có ảnh hợp lệ!")
        return [], [], None

    image_input = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    return original_images, valid_image_files, image_features


original_images, image_files, image_features = get_image_features(folder_path, device)

title = st.title("Image Search with CLIP")
# query = st.text_input("Enter text")
query = st.text_input("Enter text to search for similar images:")
if query:
    text_tokens = clip.tokenize(["This is " + query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarity = similarity.squeeze(0)
    top_k = st.slider("Select number of top similar images to display", min_value=1, max_value=30, value=5)
    top_k_indices = np.argsort(-similarity)[:top_k].tolist()
    for i in top_k_indices:
        st.write(f"Image: {image_files[i]}, Similarity: {similarity[i]:.4f}")
        st.image(original_images[i])
else:
    st.stop()