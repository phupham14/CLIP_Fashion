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

## 🛠️ Installation

### 1. Clone project

```bash
git clone https://github.com/phupham14/CLIP_Fashion
cd CLIP_Fashion
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install torch torchvision torchaudio streamlit pandas numpy pillow
```
### 3. Chuẩn bị dataset 
- Tải dataset Fashion Product Images từ Kaggle:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- Giải nén dataset, đặt file styles.csv và thư mục ảnh trong folder CLIP_Fashion

### 4. Load model
- Chạy file ```bash CLIP_Fashion.ipynb ```
- Sau khi chạy xong, file model ```bash clip_fashion_export.pt ``` được export
- Import file model vào main.py
- Bạn không cần tải lại từ OpenAI.

### 5. Chạy Streamlit app
```bash
streamlit run app.py
```
- Mở trình duyệt theo link được hiển thị trên terminal (mặc định http://localhost:8501)
- Nhập mô tả sản phẩm để tìm kiếm ảnh thời trang tương ứng.
