# 🚀 Hướng dẫn Triển khai - LawBot v8.2

## 📋 **Tổng quan**

Tài liệu này hướng dẫn cách triển khai hệ thống **LawBot v8.2** trên môi trường production. Hướng dẫn này bao gồm các bước từ cài đặt môi trường, cấu hình, huấn luyện mô hình cho đến triển khai bằng Docker.

---

## 🎯 **Yêu cầu hệ thống**

### **Môi trường khuyến nghị:**

-   **Hệ điều hành:** Ubuntu 20.04+ hoặc Windows Server 2019+
-   **Python:** 3.8+
-   **RAM:** Tối thiểu 16GB (32GB+ cho quá trình training)
-   **GPU:** NVIDIA GPU với VRAM tối thiểu 8GB (khuyến nghị cho training và inference tốc độ cao)
-   **Lưu trữ:** Tối thiểu 20GB dung lượng trống cho mô hình, dữ liệu và index.

### **Cài đặt phụ thuộc hệ thống (cho Ubuntu/Debian):**

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git
```

---

## 🔧 **Cài đặt Thủ công**

Quy trình này phù hợp cho việc chạy và thử nghiệm trực tiếp trên máy chủ.

### **1. Chuẩn bị Project**

```bash
# Clone repository
git clone <URL_CUA_REPOSITORY>
cd LawBot

# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate  # Trên Linux
# venv\Scripts\activate   # Trên Windows

# Nâng cấp pip và cài đặt các gói cần thiết
pip install --upgrade pip
pip install -r requirements.txt
```

### **2. Cấu hình Môi trường**

Hệ thống sử dụng các biến môi trường để cấu hình. Tạo một tệp `.env` ở thư mục gốc của dự án:

```sh
# Chế độ hoạt động: 'quality' (chất lượng cao) hoặc 'fast' (tốc độ nhanh)
LAWBOT_PERFORMANCE_MODE="quality"

# Bật/tắt chế độ debug
LAWBOT_DEBUG="false"

# (Tùy chọn) Chỉ định thiết bị GPU
CUDA_VISIBLE_DEVICES="0"
```

### **3. Tải Dữ liệu và Mô hình**

-   **Dữ liệu:** Đảm bảo thư mục `data/raw` chứa các tệp `legal_corpus.json`, `train.json`, và `public_test.json`.
-   **Mô hình đã huấn luyện (Tùy chọn):** Nếu bạn đã có các mô hình đã huấn luyện, hãy đặt chúng vào thư mục `models/`.

### **4. Chạy Pipeline Huấn luyện**

Nếu bạn cần huấn luyện lại các mô hình từ đầu, sử dụng script `run_pipeline.py`.

```bash
# Chạy toàn bộ pipeline từ đầu đến cuối
python run_pipeline.py

# Để tiếp tục từ một bước đã thất bại trước đó
python run_pipeline.py --resume

# Để bắt đầu từ một bước cụ thể (ví dụ: 'train_bi_encoder')
python run_pipeline.py --start-step train_bi_encoder
```

### **5. Khởi chạy Ứng dụng Streamlit**

Sau khi pipeline hoàn tất và các mô hình đã sẵn sàng, bạn có thể khởi chạy giao diện người dùng.

```bash
streamlit run app/app.py
```

---

## 🐳 **Triển khai với Docker**

Đây là phương pháp được khuyến nghị cho môi trường production ổn định.

### **1. Dockerfile**

Dưới đây là một `Dockerfile` mẫu để đóng gói ứng dụng:

```dockerfile
# Sử dụng base image Python
FROM python:3.9-slim

# Cài đặt các phụ thuộc hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép và cài đặt các gói Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của ứng dụng
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p /app/data /app/models /app/indexes /app/logs /app/reports

# Thiết lập biến môi trường
ENV LAWBOT_PERFORMANCE_MODE="quality"
ENV LAWBOT_DEBUG="false"
ENV PYTHONPATH=/app

# Expose cổng của Streamlit
EXPOSE 8501

# Lệnh để chạy ứng dụng
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **2. Docker Compose**

Sử dụng `docker-compose.yml` để quản lý container một cách dễ dàng.

```yaml
version: '3.8'

services:
  lawbot:
    build: .
    container_name: lawbot_app
    ports:
      - "8501:8501"
    volumes:
      # Mount các thư mục dữ liệu và mô hình từ máy chủ vào container
      - ./data:/app/data
      - ./models:/app/models
      - ./indexes:/app/indexes
      - ./logs:/app/logs
    environment:
      - LAWBOT_PERFORMANCE_MODE=quality
      - LAWBOT_DEBUG=false
    # Thêm cấu hình GPU nếu có
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### **3. Build và Chạy**

```bash
# Build Docker image
docker-compose build

# Khởi chạy container
docker-compose up -d
```

---

## 🛠️ **Dành cho Nhà phát triển (Development)**

### **Cài đặt Gói cho Development**

Để phát triển và đóng góp cho dự án, bạn cần cài đặt thêm các gói cho việc kiểm thử, định dạng mã và phân tích.

```bash
pip install -r requirements.txt

# Cài đặt các công cụ development
pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    isort \
    pre-commit
```

### **Chạy Kiểm thử (Tests)**

```bash
pytest
```

### **Sử dụng Pre-commit**

Dự án sử dụng `pre-commit` để đảm bảo chất lượng mã.

```bash
# Cài đặt git hooks
pre-commit install

# Chạy trên tất cả các file
pre-commit run --all-files
```

---

## 🚨 **Xử lý sự cố (Troubleshooting)**

-   **Lỗi `Out of Memory` (Hết bộ nhớ):**
    -   Thử giảm `BATCH_SIZE` trong tệp `config_quality.py` hoặc `config_fast.py`.
    -   Nếu đang training, đảm bảo không có ứng dụng nào khác chiếm dụng nhiều VRAM của GPU.
-   **Ứng dụng không khởi động:**
    -   Kiểm tra logs của Streamlit hoặc Docker để tìm lỗi chi tiết.
    -   Đảm bảo rằng tất cả các đường dẫn trong các tệp `config_*.py` là chính xác.
    -   Xác nhận rằng các mô hình và index đã được tạo thành công trong thư mục `models/` và `indexes/`.
-   **Hiệu suất chậm:**
    -   Nếu không sử dụng GPU, quá trình inference sẽ chậm hơn đáng kể. Hãy cân nhắc sử dụng GPU.
    -   Chuyển sang chế độ `fast` bằng cách đặt `LAWBOT_PERFORMANCE_MODE="fast"` trong tệp `.env` để sử dụng các mô hình và cấu hình nhẹ hơn. 