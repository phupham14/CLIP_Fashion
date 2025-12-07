# 🧥 Fashion Search using CLIP  
Tìm kiếm sản phẩm thời trang bằng mô tả văn bản, sử dụng mô hình **OpenAI CLIP** và giao diện **Streamlit**.

---

## ✨ Features

- 🔍 **Text-to-Image search** bằng CLIP  
- ⚡ Tốc độ nhanh nhờ **precomputed image features**  
- 🏷️ Hiển thị **caption** từ `styles.csv` (`productDisplayName`)  
- 🖼️ UI đẹp dạng **grid 3 cột**  
- 📦 Tương thích CPU & GPU  
- 💾 Cache embedding để load nhanh  

---

## 📁 Dataset

Dùng dataset thời trang từ Kaggle: **Fashion Product Images Dataset**  

Chứa:
- ~44k ảnh kèm theo caption
- Link dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

---

## 🧠 Model

File model đã export được đặt tại: Model/clip_fashion_export.pt


Model bao gồm:
- `model_name`: backbone CLIP  
- `state_dict`: trọng số fine-tuned  

---

## 🛠️ Installation

### 1. Clone project

```bash
git clone https://github.com/phupham14/CLIP_Fashion
cd CLIP_Fashion

