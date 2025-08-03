# 🚀 QUICK START GUIDE - LawBot

> **Hướng dẫn nhanh cho người mới bắt đầu**  
> Giải thích chi tiết theo nguyên tắc 5WH (What, Why, When, Where, Who, How)

## 📋 **TỔNG QUAN (WHAT)**

### **LawBot là gì?**
LawBot là một hệ thống hỏi-đáp pháp luật thông minh được thiết kế đặc biệt cho pháp luật Việt Nam. Hệ thống sử dụng công nghệ AI tiên tiến để trả lời các câu hỏi về pháp luật một cách chính xác và nhanh chóng.

### **Kiến trúc hệ thống:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Câu hỏi      │───▶│  Bi-Encoder     │───▶│  Top-K Results  │
│   của người    │    │  (Retrieval)    │    │  (100 docs)     │
│   dùng         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Câu trả lời  │◀───│  Cross-Encoder  │◀───│  Re-ranked      │
│   chính xác    │    │  (Reranking)    │    │  Top-5 Results  │
│   nhất         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 **TẠI SAO CẦN LAW BOT? (WHY)**

### **Vấn đề hiện tại:**
- ❌ **Khó tìm kiếm**: Văn bản pháp luật rất nhiều và phức tạp
- ❌ **Thời gian chậm**: Tìm kiếm thủ công mất nhiều thời gian
- ❌ **Thiếu chính xác**: Kết quả tìm kiếm không đúng trọng tâm
- ❌ **Khó hiểu**: Ngôn ngữ pháp lý khó hiểu với người không chuyên

### **Giải pháp LawBot:**
- ✅ **Tìm kiếm nhanh**: AI tìm kiếm trong vài giây
- ✅ **Kết quả chính xác**: Sử dụng Cross-Encoder để đánh giá độ liên quan
- ✅ **Dễ hiểu**: Trả về văn bản pháp luật có liên quan nhất
- ✅ **Tiết kiệm thời gian**: Từ vài giờ xuống còn vài giây

## ⏰ **KHI NÀO SỬ DỤNG? (WHEN)**

### **Các trường hợp sử dụng:**
- 🔍 **Tìm kiếm luật**: "Người lao động được nghỉ phép bao nhiêu ngày?"
- 📋 **Tra cứu quy định**: "Điều kiện thành lập doanh nghiệp là gì?"
- ⚖️ **So sánh văn bản**: "Sự khác biệt giữa Luật Doanh nghiệp cũ và mới?"
- 📝 **Tìm điều khoản**: "Điều 113 Bộ luật Lao động quy định gì?"

### **Khi nào KHÔNG sử dụng:**
- ❌ Câu hỏi không liên quan đến pháp luật
- ❌ Yêu cầu tư vấn pháp lý chuyên sâu
- ❌ Thay thế hoàn toàn luật sư

## 📍 **SỬ DỤNG Ở ĐÂU? (WHERE)**

### **Môi trường phát triển:**
- 💻 **Local Development**: Máy tính cá nhân
- 🖥️ **Server**: Máy chủ công ty
- ☁️ **Cloud**: AWS, Google Cloud, Azure
- 🐳 **Docker**: Container deployment

### **Yêu cầu hệ thống:**
```
OS: Windows 10+, Ubuntu 18.04+, macOS 10.14+
RAM: Tối thiểu 8GB, khuyến nghị 16GB+
Storage: Tối thiểu 10GB cho models và data
GPU: NVIDIA GPU với CUDA (khuyến nghị)
Python: 3.8+
```

## 👥 **AI CHO? (WHO)**

### **Đối tượng sử dụng:**
- 👨‍💼 **Luật sư**: Tra cứu nhanh văn bản pháp luật
- 👨‍💻 **Nhà phát triển**: Tích hợp vào ứng dụng pháp lý
- 👨‍🎓 **Sinh viên luật**: Học tập và nghiên cứu
- 👨‍💼 **Doanh nghiệp**: Tuân thủ quy định pháp luật
- 👨‍👩‍👧‍👦 **Công dân**: Tìm hiểu quyền và nghĩa vụ

### **Đối tượng phát triển:**
- 👨‍💻 **AI/ML Engineers**: Phát triển và tối ưu models
- 👨‍💻 **Software Engineers**: Tích hợp và deployment
- 👨‍💻 **Data Scientists**: Phân tích và cải thiện performance
- 👨‍💻 **DevOps Engineers**: Deployment và monitoring

## 🔧 **LÀM THẾ NÀO? (HOW)**

### **Bước 1: Chuẩn bị môi trường**

#### **1.1 Cài đặt Python**
```bash
# Kiểm tra Python version
python --version
# Kết quả mong đợi: Python 3.8.0 hoặc cao hơn

# Nếu chưa có Python, tải từ python.org
```

#### **1.2 Clone repository**
```bash
# Clone project về máy
git clone https://github.com/lawbot-team/lawbot.git
cd lawbot

# Kiểm tra cấu trúc thư mục
ls -la
# Kết quả mong đợi: thấy các thư mục core/, scripts/, data/, etc.
```

#### **1.3 Tạo virtual environment**
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate

# Kiểm tra đã kích hoạt thành công
# Terminal sẽ hiển thị (venv) ở đầu dòng
```

#### **1.4 Cài đặt dependencies**
```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Kiểm tra cài đặt thành công
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
```

### **Bước 2: Chuẩn bị dữ liệu**

#### **2.1 Kiểm tra dữ liệu**
```bash
# Kiểm tra thư mục data
ls data/raw/
# Kết quả mong đợi: legal_corpus.json, train.json

# Kiểm tra kích thước file
du -h data/raw/legal_corpus.json
du -h data/raw/train.json
```

#### **2.2 Tạo thư mục cần thiết**
```bash
# Tạo các thư mục cho models và indexes
mkdir -p models indexes reports logs

# Kiểm tra đã tạo thành công
ls -la
# Kết quả mong đợi: thấy các thư mục models/, indexes/, reports/, logs/
```

### **Bước 3: Kiểm tra môi trường**

#### **3.1 Chạy script kiểm tra**
```bash
# Chạy script kiểm tra môi trường
python scripts/01_check_environment.py

# Kết quả mong đợi:
# ✅ Python version: 3.8.0
# ✅ CUDA available: True/False
# ✅ Required packages installed
# ✅ Data files found
# ✅ Directories created
```

#### **3.2 Xử lý lỗi nếu có**
```bash
# Nếu thiếu package nào đó
pip install tên_package

# Nếu thiếu file data
# Tải file data từ nguồn chính thức

# Nếu lỗi CUDA
# Cài đặt CUDA toolkit hoặc sử dụng CPU
```

### **Bước 4: Chạy pipeline**

#### **4.1 Chạy toàn bộ pipeline**
```bash
# Chạy pipeline hoàn chỉnh
python run_pipeline.py

# Kết quả mong đợi:
# [START] BAT DAU LEGAL QA PIPELINE
# [OK] Buoc 01: Kiem tra Moi truong
# [OK] Buoc 02: Loc Dataset Chat luong
# [OK] Buoc 03: Tien xu ly Du lieu
# ...
# [SUCCESS] PIPELINE HOAN THANH!
```

#### **4.2 Chạy từng bước riêng lẻ**
```bash
# Chạy từ bước cụ thể
python run_pipeline.py --step 03

# Bỏ qua filtering
python run_pipeline.py --skip-filtering

# Xem danh sách bước
python run_pipeline.py --list-steps
```

### **Bước 5: Sử dụng API**

#### **5.1 Tạo script test**
```python
# Tạo file test_api.py
from core.pipeline import LegalQAPipeline

# Khởi tạo pipeline
pipeline = LegalQAPipeline()

# Kiểm tra trạng thái
if pipeline.is_ready:
    print("✅ Pipeline sẵn sàng!")
else:
    print("❌ Pipeline chưa sẵn sàng!")
    exit(1)

# Test câu hỏi
query = "Người lao động được nghỉ phép bao nhiêu ngày?"
print(f"🔍 Câu hỏi: {query}")

# Thực hiện dự đoán
results = pipeline.predict(
    query=query,
    top_k_retrieval=100,
    top_k_final=5
)

# In kết quả
print(f"\n📋 Tìm thấy {len(results)} kết quả:")
for i, result in enumerate(results, 1):
    print(f"\n--- Kết quả {i} ---")
    print(f"📄 AID: {result['aid']}")
    print(f"⭐ Điểm: {result['rerank_score']:.4f}")
    print(f"📝 Nội dung: {result['content'][:200]}...")
```

#### **5.2 Chạy test**
```bash
# Chạy script test
python test_api.py

# Kết quả mong đợi:
# ✅ Pipeline sẵn sàng!
# 🔍 Câu hỏi: Người lao động được nghỉ phép bao nhiêu ngày?
# 📋 Tìm thấy 5 kết quả:
# --- Kết quả 1 ---
# 📄 AID: law_1_113
# ⭐ Điểm: 0.9234
# 📝 Nội dung: Điều 113. Người lao động được nghỉ phép năm...
```

### **Bước 6: Monitoring và Debug**

#### **6.1 Kiểm tra logs**
```bash
# Xem logs mới nhất
tail -f logs/pipeline_*.log

# Tìm lỗi trong logs
grep "ERROR" logs/pipeline_*.log

# Xem thống kê
grep "SUCCESS\|FAIL" logs/pipeline_*.log
```

#### **6.2 Kiểm tra performance**
```bash
# Kiểm tra sử dụng GPU
nvidia-smi

# Kiểm tra sử dụng memory
free -h

# Kiểm tra CPU usage
top
```

## 📊 **INPUT/OUTPUT CHI TIẾT**

### **Input:**
```
Câu hỏi: "Người lao động được nghỉ phép bao nhiêu ngày?"
```

### **Output:**
```json
[
  {
    "aid": "law_1_113",
    "content": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
    "retrieval_score": 0.85,
    "rerank_score": 0.92
  },
  {
    "aid": "law_1_114", 
    "content": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc...",
    "retrieval_score": 0.78,
    "rerank_score": 0.87
  }
]
```

## 🚨 **TROUBLESHOOTING**

### **Lỗi thường gặp:**

#### **1. "ModuleNotFoundError: No module named 'config'"**
```bash
# Nguyên nhân: Python path không đúng
# Giải pháp:
export PYTHONPATH="${PYTHONPATH}:/path/to/lawbot"
# Hoặc chạy từ project root
cd /path/to/lawbot
python scripts/01_check_environment.py
```

#### **2. "CUDA out of memory"**
```bash
# Nguyên nhân: GPU memory không đủ
# Giải pháp:
export LAWBOT_BI_ENCODER_BATCH_SIZE=2
export LAWBOT_CROSS_ENCODER_BATCH_SIZE=1
# Hoặc sử dụng CPU
export CUDA_VISIBLE_DEVICES=""
```

#### **3. "File not found: data/raw/legal_corpus.json"**
```bash
# Nguyên nhân: File dữ liệu chưa được tải
# Giải pháp:
ls -la data/raw/
# Tải file data nếu cần
```

## 📈 **METRICS VÀ ĐÁNH GIÁ**

### **Performance Metrics:**
```
Precision@5: 75-85%
Recall@5: 65-75%
F1@5: 70-80%
MRR: 0.7-0.8
Response Time: 250-600ms
```

### **Kiểm tra metrics:**
```bash
# Chạy evaluation
python scripts/12_evaluate_pipeline.py

# Xem kết quả
cat reports/evaluation_report_*.json
```

## 🎯 **NEXT STEPS**

### **Sau khi hoàn thành Quick Start:**
1. 📖 **Đọc README.md** - Hiểu chi tiết về hệ thống
2. 🔧 **Tùy chỉnh config** - Điều chỉnh theo nhu cầu
3. 🚀 **Deploy production** - Triển khai lên server
4. 📊 **Monitor performance** - Theo dõi hiệu suất
5. 🔄 **Cải thiện models** - Fine-tune theo dữ liệu mới

---

**🎉 Chúc mừng! Bạn đã hoàn thành Quick Start với LawBot! 🚀** 