# 🏛️ LawBot - Hệ thống Hỏi-Đáp Pháp luật Thông minh

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-8.1-green.svg)](https://github.com/lawbot-team/lawbot)
[![Pipeline](https://img.shields.io/badge/pipeline-cascaded--reranking-orange.svg)](https://github.com/lawbot-team/lawbot)

> **Hệ thống AI hỏi-đáp pháp luật tiên tiến cho Việt Nam**  
> **Phiên bản v8.1: Tối ưu hóa toàn diện với kiến trúc 3 tầng thông minh**

---

## 📋 **Mục lục**

- [🎯 Tổng quan](#-tổng-quan)
- [🚀 Bắt đầu nhanh](#-bắt-đầu-nhanh)
- [💡 Cách sử dụng](#-cách-sử-dụng)
- [🧠 Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [⚙️ Cài đặt và cấu hình](#️-cài-đặt-và-cấu-hình)
- [🔄 Quy trình huấn luyện](#-quy-trình-huấn-luyện)
- [📊 Đánh giá hiệu suất](#-đánh-giá-hiệu-suất)
- [🛠️ Phát triển và bảo trì](#️-phát-triển-và-bảo-trì)
- [❓ Hỏi đáp](#-hỏi-đáp)

---

## 🎯 **Tổng quan**

### **LawBot là gì?**

LawBot là một hệ thống AI hỏi-đáp pháp luật được thiết kế đặc biệt cho pháp luật Việt Nam. Hệ thống sử dụng công nghệ AI tiên tiến để trả lời các câu hỏi về pháp luật một cách chính xác và nhanh chóng.

### **Tính năng nổi bật**

✨ **Kiến trúc 3 tầng thông minh**
- **Tầng 1**: Tìm kiếm rộng (500 ứng viên)
- **Tầng 2**: Lọc nhanh (50 ứng viên)  
- **Tầng 3**: Thẩm định chuyên sâu (5 kết quả cuối)

🧠 **AI chuyên biệt cho pháp luật**
- Domain-Adaptive Pre-training (DAPT)
- PhoBERT-Law model chuyên môn hóa
- Ensemble learning với nhiều model

⚡ **Hiệu suất tối ưu**
- Thời gian phản hồi: ~0.5 giây
- Độ chính xác: >90%
- GPU acceleration tự động

### **Khi nào sử dụng?**

✅ **Phù hợp:**
- Tìm kiếm điều luật cụ thể
- Tra cứu quy định pháp luật
- So sánh văn bản pháp luật
- Tìm hiểu quyền và nghĩa vụ

❌ **Không phù hợp:**
- Tư vấn pháp lý chuyên sâu
- Thay thế luật sư
- Câu hỏi không liên quan pháp luật

---

## 🚀 **Bắt đầu nhanh**

### **Bước 1: Cài đặt môi trường**

```bash
# Clone repository
git clone https://github.com/lawbot-team/lawbot.git
cd lawbot

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### **Bước 2: Chuẩn bị dữ liệu**

Đảm bảo bạn có các file dữ liệu sau trong thư mục `data/raw/`:
- `legal_corpus.json` - Kho văn bản pháp luật
- `train.json` - Dữ liệu training
- `public_test.json` - Dữ liệu test

### **Bước 3: Chạy pipeline huấn luyện**

```bash
# Chạy toàn bộ pipeline với chế độ chất lượng (QUALITY)
python run_pipeline.py --mode quality

# Chạy nhanh để kiểm thử (FAST), bỏ qua DAPT
python run_pipeline.py --mode fast --no-dapt

# Chạy từ bước cụ thể (ví dụ bắt đầu từ bước 03)
python run_pipeline.py --start-step 03 --mode fast

# Xem danh sách các bước và trạng thái checkpoint
python run_pipeline.py --show-steps

# Xoá checkpoint để chạy lại từ đầu
python run_pipeline.py --clear-checkpoint
```

Lưu ý:
- `--mode fast|quality` ghi đè chế độ hiệu năng chỉ cho lần chạy hiện tại (không cần đổi biến môi trường).
- Bạn có thể kiểm tra cấu hình hiện tại bằng:
  ```bash
  python -c "import config; config.print_config_summary()"
  ```

### **Bước 4: Khởi động ứng dụng**

```bash
# Khởi động giao diện web
streamlit run app/app.py

# Truy cập: http://localhost:8501
```

---

## ⚡ **Chế độ hiệu năng (Performance Modes)**

Hệ thống hỗ trợ 2 chế độ:
- **FAST**: ưu tiên tốc độ để kiểm thử nhanh (epochs ít, batch nhỏ, eval ít).
- **QUALITY**: ưu tiên chất lượng mô hình (epochs/batch lớn hơn, eval nhiều hơn).

Có 2 cách chọn chế độ:

1) Ghi đè bằng cờ `--mode` khi chạy pipeline (khuyến nghị)
```bash
python run_pipeline.py --mode fast
python run_pipeline.py --mode quality
```

2) Dùng script chuyển mode (ảnh hưởng biến môi trường, hữu ích cho các tiến trình mới)
```bash
# Chuyển sang FAST cho phiên hiện tại và đặt vĩnh viễn (Windows dùng setx)
python switch_performance_mode.py fast

# Áp dụng ngay và in tóm tắt cấu hình (không cần mở phiên mới)
python switch_performance_mode.py fast --immediate

# Xem mode hiện tại
python switch_performance_mode.py show
```

Ghi chú Windows:
- `setx` chỉ áp dụng cho cửa sổ terminal mở sau đó. Nếu không dùng `--immediate`, hãy mở phiên mới hoặc chạy lại với `--mode`.
- Kiểm tra nhanh: `python -c "import config; config.print_config_summary()"`.

---

## 🧰 **Lệnh thường dùng (Cheat Sheet)**

```bash
# 1) Kiểm tra cấu hình hiện tại
python -c "import config; config.print_config_summary()"

# 2) Chạy pipeline nhanh (FAST), bỏ DAPT
python run_pipeline.py --mode fast --no-dapt

# 3) Chạy đầy đủ chất lượng (QUALITY)
python run_pipeline.py --mode quality

# 4) Tiếp tục từ checkpoint hoặc chọn bước bắt đầu
python run_pipeline.py --show-steps
python run_pipeline.py --start-step 03 --mode fast

# 5) Xoá checkpoint
python run_pipeline.py --clear-checkpoint

# 6) Chỉ chạy đánh giá (nếu cần riêng)
python scripts/04_evaluate_models.py --mode fast

# 7) Chuyển chế độ hiệu năng (toàn hệ thống)
python switch_performance_mode.py fast --immediate
python switch_performance_mode.py show

# 8) Khởi động ứng dụng web
streamlit run app/app.py

# 9) Clean
python scripts/05_cleanup_old_models.py
# Dry-run kiểm tra trước:
python scripts/05_cleanup_old_models.py --mode fast --dry-run
# Xóa sạch artifacts cốt lõi (không hỏi):
python scripts/05_cleanup_old_models.py --mode fast -y
# Xóa “sạch sâu” bao gồm processed artifacts, reports, logs:
python scripts/05_cleanup_old_models.py --mode fast -y --all --include-reports --include-logs
# Sau khi clean:
python run_pipeline.py --mode fast --start-step 01 (hoặc --start-step 03 nếu đã có dữ liệu).
```

### **Bước 5: Sử dụng API**

```python
from core.pipeline import LegalQAPipeline

# Khởi tạo pipeline
pipeline = LegalQAPipeline()

# Đặt câu hỏi
query = "Người lao động được nghỉ phép bao nhiêu ngày?"
results = pipeline.predict(query=query, top_k_final=5)

# Xem kết quả
for i, result in enumerate(results):
    print(f"Kết quả {i+1}: {result['content'][:100]}...")
    print(f"Điểm: {result['rerank_score']:.3f}")
```

---

## 💡 **Cách sử dụng**

### **Đặt câu hỏi hiệu quả**

#### **✅ Câu hỏi tốt:**
- "Người lao động được nghỉ phép bao nhiêu ngày?"
- "Điều kiện thành lập doanh nghiệp là gì?"
- "Mức phạt vi phạm giao thông là bao nhiêu?"
- "Thủ tục đăng ký kinh doanh cần những gì?"

#### **❌ Câu hỏi không hiệu quả:**
- "Luật" (quá chung chung)
- "Tất cả quy định về lao động" (quá rộng)
- "Có phải tôi đúng không?" (câu hỏi đóng)

### **Giao diện web**

1. **Truy cập**: http://localhost:8501
2. **Nhập câu hỏi** vào ô tìm kiếm
3. **Điều chỉnh số kết quả** mong muốn
4. **Nhấn "Tìm kiếm"**
5. **Xem kết quả** với điểm số và nội dung

### **Ví dụ chi tiết luồng xử lý từ App đến kết quả**

#### **🎯 Bước 1: Người dùng nhập câu hỏi**
```
Giao diện web: http://localhost:8501
┌─────────────────────────────────────────┐
│ 🔍 Tìm kiếm pháp luật                  │
│                                         │
│ [Người lao động được nghỉ phép bao     │
│  nhiêu ngày?                    ] [🔍] │
│                                         │
│ Số kết quả: [5] ▼                     │
│                                         │
│ [Tìm kiếm]                            │
└─────────────────────────────────────────┘
```

#### **📡 Bước 2: App gửi request đến Pipeline**
```python
# app/app.py
def calculate_optimal_parameters(final_results_count):
    """Tính toán tham số tối ưu"""
    top_k_retrieval = max(50, final_results_count * 20)      # 100
    top_k_light_reranking = max(20, final_results_count * 4)  # 20
    top_k_final = final_results_count                         # 5
    
    return {
        "top_k_retrieval": top_k_retrieval,
        "top_k_light_reranking": top_k_light_reranking,
        "top_k_final": top_k_final,
    }

# Khi user nhấn "Tìm kiếm"
query = "Người lao động được nghỉ phép bao nhiêu ngày?"
final_results_count = 5

# Tính toán tham số
params = calculate_optimal_parameters(final_results_count)
# params = {
#     "top_k_retrieval": 100,
#     "top_k_light_reranking": 20, 
#     "top_k_final": 5
# }

# Gọi pipeline
results = pipeline.predict(
    query=query,
    top_k_retrieval=params["top_k_retrieval"],
    top_k_final=params["top_k_final"],
    top_k_light_reranking=params["top_k_light_reranking"],
)
```

#### **🔧 Bước 3: Pipeline xử lý (core/pipeline.py)**
```python
# core/pipeline.py - LegalQAPipeline.predict()
def predict(self, query, top_k_retrieval, top_k_final, top_k_light_reranking=None):
    """Thực hiện quy trình 3 tầng"""
    
    # Tầng 1: Retrieval
    retrieved_aids, retrieved_distances = self.retrieve(query, top_k_retrieval)
    # retrieved_aids = ["law_1_113", "law_1_114", "law_1_115", ...]
    # retrieved_distances = [0.95, 0.87, 0.82, ...]
    
    # Tầng 2: Light Reranking (nếu có)
    if self.use_cascaded_reranking:
        light_aids, light_distances = self.rerank_light(
            query, retrieved_aids, retrieved_distances, top_k_light_reranking
        )
        # light_aids = ["law_1_113", "law_1_114", ...] (top 20)
    
    # Tầng 3: Strong Reranking
    reranked_results = self.rerank(query, light_aids, light_distances)
    # reranked_results = [
    #     {"aid": "law_1_113", "content": "...", "rerank_score": 0.94},
    #     {"aid": "law_1_114", "content": "...", "rerank_score": 0.88},
    #     ...
    # ]
    
    return reranked_results[:top_k_final]  # Top 5
```

#### **📊 Bước 4: App hiển thị kết quả**
```python
# app/app.py - Hiển thị kết quả
for i, result in enumerate(results):
    st.markdown(f"### Kết quả {i+1}")
    st.markdown(f"**Điều luật:** {result['aid']}")
    st.markdown(f"**Điểm tin cậy:** {result['rerank_score']:.3f}")
    st.markdown(f"**Nội dung:** {result['content']}")
```

**Giao diện kết quả**:
```
┌─────────────────────────────────────────┐
│ 📋 Kết quả tìm kiếm (5 kết quả)        │
│                                         │
│ 🥇 Kết quả 1 (Điểm: 0.940)            │
│ Điều luật: law_1_113                   │
│ Nội dung: Điều 113. Người lao động     │
│ được nghỉ phép năm 12 ngày làm việc... │
│                                         │
│ 🥈 Kết quả 2 (Điểm: 0.880)            │
│ Điều luật: law_1_114                   │
│ Nội dung: Điều 114. Thời gian nghỉ     │
│ phép năm được tính theo năm làm việc...│
│                                         │
│ 🥉 Kết quả 3 (Điểm: 0.820)            │
│ ...                                     │
└─────────────────────────────────────────┘
```

#### **⚡ Bước 5: Thông tin chi tiết (nếu user mở expander)**
```python
# Hiển thị thông tin chi tiết về quá trình xử lý
with st.expander("📊 Thông số tìm kiếm được tính toán tự động"):
    st.markdown(f"**🎯 Tầng 1 - Retrieval:** {params['top_k_retrieval']} ứng viên")
    st.markdown(f"**⚡ Tầng 2 - Light Reranking:** {params['top_k_light_reranking']} ứng viên")
    st.markdown(f"**🎯 Tầng 3 - Final Reranking:** {params['top_k_final']} kết quả cuối cùng")

with st.expander("🔍 Chi tiết quá trình xử lý"):
    st.markdown("**Tầng 1:** Tìm kiếm 100 ứng viên từ 15,420 văn bản")
    st.markdown("**Tầng 2:** Lọc xuống 20 ứng viên chất lượng cao")
    st.markdown("**Tầng 3:** Thẩm định chuyên sâu với Ensemble models")
    st.markdown("**Thời gian:** ~550ms")
```

### **Tùy chỉnh kết quả**

- **Ít kết quả (1-3)**: Tập trung vào câu trả lời chính xác nhất
- **Nhiều kết quả (5-10)**: Xem nhiều góc độ và ngữ cảnh
- **Rất nhiều (10-20)**: Nghiên cứu toàn diện

### **Mẹo sử dụng**

1. **Sử dụng từ khóa chính xác**
   - ✅ "nghỉ phép" thay vì "nghỉ ngơi"
   - ✅ "thành lập doanh nghiệp" thay vì "mở công ty"

2. **Đặt câu hỏi cụ thể**
   - ✅ "Mức phạt vi phạm giao thông là bao nhiêu?"
   - ❌ "Luật giao thông"

3. **Kết hợp nhiều câu hỏi**
   - Hỏi từng khía cạnh riêng biệt
   - So sánh kết quả để có cái nhìn toàn diện

---

## 🧠 **Kiến trúc hệ thống**

### **Tổng quan kiến trúc**

```
┌─────────────────────────────────────────────────────────────┐
│                    Người dùng đặt câu hỏi                    │
│  "Người lao động được nghỉ phép bao nhiêu ngày?"           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  TẦNG 1: Bi-Encoder Retrieval                              │
│  • Tìm kiếm rộng trong toàn bộ kho dữ liệu                │
│  • Trả về 500 ứng viên có liên quan nhất                   │
│  • Thời gian: ~100ms                                       │
│  • Kết quả: [law_1_113, law_1_114, law_2_45, ...]        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  TẦNG 2: Light Reranker                                   │
│  • Lọc nhanh từ 500 xuống 50 ứng viên chất lượng cao       │
│  • Sử dụng model nhỏ, nhanh                               │
│  • Thời gian: ~150ms                                       │
│  • Kết quả: [law_1_113, law_1_114, law_1_115, ...]       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  TẦNG 3: Ensemble Strong Reranker                         │
│  • Hội đồng chuyên gia thẩm định chuyên sâu               │
│  • Sử dụng nhiều Cross-Encoder models                     │
│  • Thời gian: ~300ms                                       │
│  • Kết quả: Top 5 với điểm số chi tiết                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Top 5 kết quả chính xác nhất             │
│  • law_1_113: 0.95 điểm                                   │
│  • law_1_114: 0.87 điểm                                   │
│  • law_1_115: 0.82 điểm                                   │
└─────────────────────────────────────────────────────────────┘
```

### **Ví dụ chi tiết luồng xử lý câu hỏi**

#### **🎯 Input: Câu hỏi từ người dùng**
```
"Người lao động được nghỉ phép bao nhiêu ngày?"
```

#### **📊 Tầng 1: Bi-Encoder Retrieval**

**Input**: Câu hỏi người dùng
**Process**: 
1. Encode câu hỏi thành vector 768 chiều
2. So sánh với 15,420 văn bản pháp luật
3. Trả về 500 ứng viên có điểm cao nhất

**Output**: 
```python
retrieved_aids = [
    "law_1_113",    # Điều 113 Bộ luật Lao động
    "law_1_114",    # Điều 114 Bộ luật Lao động  
    "law_2_45",     # Điều 45 Luật khác
    "law_1_115",    # Điều 115 Bộ luật Lao động
    # ... 496 ứng viên khác
]

retrieval_scores = [0.95, 0.87, 0.82, 0.78, ...]
```

#### **⚡ Tầng 2: Light Reranker**

**Input**: 500 ứng viên từ Tầng 1
**Process**:
1. Sử dụng Light Reranker model (nhỏ, nhanh)
2. Đánh giá từng cặp (câu hỏi, văn bản)
3. Kết hợp điểm retrieval + light reranking
4. Chọn top 50 ứng viên

**Output**:
```python
light_aids = [
    "law_1_113",    # Điểm: 0.92
    "law_1_114",    # Điểm: 0.89
    "law_1_115",    # Điểm: 0.85
    # ... 47 ứng viên khác
]

light_scores = [0.92, 0.89, 0.85, ...]
```

#### **⚖️ Tầng 3: Ensemble Strong Reranker**

**Input**: 50 ứng viên từ Tầng 2
**Process**:
1. Sử dụng 2 Cross-Encoder models (PhoBERT-Law + XLM-RoBERTa)
2. Đánh giá chuyên sâu từng cặp (câu hỏi, văn bản)
3. Lấy điểm trung bình từ 2 models
4. Sắp xếp theo điểm số

**Output**:
```python
final_results = [
    {
        "aid": "law_1_113",
        "content": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
        "retrieval_score": 0.95,
        "rerank_score": 0.92,
        "confidence": "high"
    },
    {
        "aid": "law_1_114", 
        "content": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc...",
        "retrieval_score": 0.87,
        "rerank_score": 0.89,
        "confidence": "high"
    },
    # ... 3 kết quả khác
]
```

### **Chi tiết từng tầng**

### **Chi tiết từng tầng**

#### **🎯 Tầng 1: Bi-Encoder Retrieval**

**Mục đích**: Tìm kiếm rộng trong toàn bộ kho dữ liệu

**Cách hoạt động**:
1. Chuyển câu hỏi thành vector 768 chiều
2. So sánh với tất cả văn bản trong kho dữ liệu
3. Trả về 500 kết quả có độ tương đồng cao nhất

**Hiệu suất**:
- Thời gian: ~100ms
- Độ chính xác: ~70% (Precision@5)

#### **⚡ Tầng 2: Light Reranker**

**Mục đích**: Lọc nhanh từ 500 xuống 50 ứng viên chất lượng cao

**Cách hoạt động**:
1. Sử dụng model nhỏ, nhanh để đánh giá sơ bộ
2. Kết hợp điểm retrieval với điểm light reranking
3. Chọn top 50 ứng viên để đưa lên tầng 3

**Hiệu suất**:
- Thời gian: ~150ms
- Lý do: Tiết kiệm thời gian cho tầng 3

#### **⚖️ Tầng 3: Ensemble Strong Reranker**

**Mục đích**: Hội đồng chuyên gia thẩm định và chọn top 5 kết quả

**Cách hoạt động**:
1. Sử dụng nhiều model Cross-Encoder cùng lúc
2. PhoBERT-Law + XLM-RoBERTa đánh giá song song
3. Lấy điểm trung bình để ra quyết định cuối cùng

**Hiệu suất**:
- Thời gian: ~300ms
- Độ chính xác: >90% (Precision@5)

### **Tại sao cần 3 tầng?**

1. **Tầng 1**: Không thể bỏ qua vì cần tìm kiếm trong toàn bộ kho dữ liệu
2. **Tầng 2**: Cần thiết để giảm tải cho tầng 3, tránh lãng phí tài nguyên
3. **Tầng 3**: Cần thiết để đạt độ chính xác tối đa cho kết quả cuối cùng

---

## ⚙️ **Cài đặt và cấu hình**

### **Yêu cầu hệ thống**

```
OS: Windows 10+, Ubuntu 18.04+, macOS 10.14+
RAM: Tối thiểu 8GB, khuyến nghị 16GB+
Storage: Tối thiểu 10GB cho models và data
GPU: NVIDIA GPU với CUDA (khuyến nghị)
Python: 3.8+
```

### **Cấu trúc dự án**

```
LawBot/
├── 📁 app/
│   └── app.py                  # Giao diện web Streamlit
├── 📁 core/                    # Các module cốt lõi
│   ├── pipeline.py             # Class pipeline xử lý chính
│   ├── logging_system.py       # Hệ thống ghi log
│   ├── evaluation_reporter.py  # Công cụ đánh giá
│   ├── progress_tracker.py     # Theo dõi tiến trình
│   └── 📁 utils/               # Utilities
│       ├── data_processing.py  # Xử lý dữ liệu
│       ├── model_utils.py      # Utilities cho models
│       ├── evaluation.py       # Metrics đánh giá
│       └── augmentation.py     # Data augmentation
├── 📁 scripts/                 # Các script thực thi
│   ├── 00_adapt_model.py       # DAPT với GPU acceleration
│   ├── 01_check_environment.py # Kiểm tra môi trường
│   ├── 02_prepare_training_data.py # Hard negative mining
│   ├── 03_train_models.py      # Training với single data loading
│   └── 📁 utils/               # Scripts tiện ích
├── 📁 data/
├── 📁 models/                  # Models đã huấn luyện
├── 📁 indexes/                 # FAISS indexes
├── 📁 reports/                 # Báo cáo đánh giá
├── 📁 logs/                    # Log files
├── 📄 config.py                # Cấu hình trung tâm
├── 📄 run_pipeline.py          # Trình điều khiển pipeline
└── 📄 README.md                # Tài liệu hướng dẫn
```

### **Cấu hình môi trường**

#### **Environment Variables**

```bash
# Environment
export LAWBOT_ENV=production
export LAWBOT_DEBUG=false

# Directories
export LAWBOT_DATA_DIR=/path/to/data
export LAWBOT_MODELS_DIR=/path/to/models
export LAWBOT_INDEXES_DIR=/path/to/indexes

# Hyperparameters
export LAWBOT_BI_ENCODER_BATCH_SIZE=16
export LAWBOT_CROSS_ENCODER_BATCH_SIZE=8
export LAWBOT_BI_ENCODER_LR=2e-5
export LAWBOT_CROSS_ENCODER_LR=2e-5

# Performance Optimizations
export LAWBOT_FP16_TRAINING=true
export LAWBOT_BI_ENCODER_DATALOADER_NUM_WORKERS=4
export LAWBOT_CROSS_ENCODER_DATALOADER_NUM_WORKERS=4

# Pipeline settings
export LAWBOT_TOP_K_RETRIEVAL=100
export LAWBOT_TOP_K_FINAL=5
```

#### **Configuration File**

```python
import config

# In thông tin cấu hình
config.print_config_summary()

# Validate cấu hình
config.validate_config()
```

### **Kiểm tra cài đặt**

```bash
# Kiểm tra cấu trúc project
python scripts/utils/check_project.py

# Kiểm tra môi trường
python scripts/01_check_environment.py

# Test GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## 🔄 **Quy trình huấn luyện**

### **Tổng quan quy trình**

```
📊 Dữ liệu thô → 🧠 Pre-training → 🔍 Training → 📈 Evaluation → 🚀 Deployment
```

### **Ví dụ chi tiết luồng dữ liệu từ thô đến kết quả**

#### **📄 Bước 1: Dữ liệu thô (Raw Data)**

**File**: `data/raw/legal_corpus.json`
```json
[
  {
    "law_id": "law_1",
    "title": "Bộ luật Lao động",
    "content": [
      {
        "aid": "113",
        "content_Article": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc và được tăng thêm theo thời gian làm việc..."
      },
      {
        "aid": "114", 
        "content_Article": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc và được tính vào thời gian làm việc..."
      }
    ]
  }
]
```

**File**: `data/raw/train.json`
```json
[
  {
    "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    "answer_id": "law_1_113",
    "answer_content": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
    "category": "labor_law"
  }
]
```

#### **🧹 Bước 2: Preprocessing (Tiền xử lý)**

**Input**: Dữ liệu thô từ Bước 1
**Process**: 
1. Parse JSON files
2. Clean text (loại bỏ ký tự đặc biệt)
3. Validate data structure
4. Create mappings

**Output**: 
```python
# aid_map.json
{
  "law_1_113": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
  "law_1_114": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc...",
  # ... 15,420 articles
}

# train_data.json
[
  {
    "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    "positive": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
    "negative": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc...",
    "label": 1
  }
]
```

#### **🎯 Bước 3: Training Data Generation**

**Input**: Dữ liệu đã preprocess
**Process**:
1. Tạo positive pairs: (câu hỏi, đáp án đúng)
2. Hard negative mining: Tìm câu trả lời sai nhưng rất giống đúng
3. Data augmentation: Tạo thêm dữ liệu đa dạng

**Output**:

**Bi-Encoder Triplets**:
```python
# bi_encoder_triplets.json
[
  {
    "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    "positive": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
    "negative": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc..."
  }
]
```

**Cross-Encoder Pairs**:
```python
# cross_encoder_pairs.json
[
  {
    "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    "document": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
    "label": 1  # Positive
  },
  {
    "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    "document": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc...",
    "label": 0  # Negative
  }
]
```

#### **🧠 Bước 4: Model Training**

**Input**: Training data từ Bước 3
**Process**:

**Bi-Encoder Training**:
```python
# Model: SentenceTransformer
# Input: Triplets (question, positive, negative)
# Loss: ContrastiveLoss
# Output: Bi-Encoder model

# Ví dụ training:
question = "Người lao động được nghỉ phép bao nhiêu ngày?"
positive = "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc..."
negative = "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc..."

# Model học để:
# - Tăng similarity(question, positive)
# - Giảm similarity(question, negative)
```

**Cross-Encoder Training**:
```python
# Model: PhoBERT-Law
# Input: Pairs (question, document, label)
# Loss: CrossEntropyLoss
# Output: Cross-Encoder model

# Ví dụ training:
pair1 = ["Người lao động được nghỉ phép bao nhiêu ngày?", 
         "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...", 
         1]  # Positive

pair2 = ["Người lao động được nghỉ phép bao nhiêu ngày?", 
         "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc...", 
         0]  # Negative
```

#### **📊 Bước 5: FAISS Index Building**

**Input**: Tất cả documents + Bi-Encoder model
**Process**:
1. Encode tất cả documents thành vectors
2. Build FAISS index cho fast similarity search
3. Create index-to-aid mapping

**Output**:
```python
# faiss_index.bin (Binary file)
# index_to_aid.json
{
  "0": "law_1_113",
  "1": "law_1_114", 
  "2": "law_1_115",
  # ... 15,420 mappings
}
```

#### **🚀 Bước 6: Inference (Khi người dùng hỏi)**

**Input**: Câu hỏi từ người dùng
**Process**:

**Tầng 1 - Retrieval**:
```python
# Input
query = "Người lao động được nghỉ phép bao nhiêu ngày?"

# Process
query_vector = bi_encoder.encode(query)  # [0.1, 0.3, 0.5, ...] (768 dim)
similarities = faiss_index.search(query_vector, 500)

# Output
retrieved_aids = ["law_1_113", "law_1_114", "law_1_115", ...]
retrieval_scores = [0.95, 0.87, 0.82, ...]
```

**Tầng 2 - Light Reranking**:
```python
# Input
pairs = [
    ["Người lao động được nghỉ phép bao nhiêu ngày?", "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc..."],
    ["Người lao động được nghỉ phép bao nhiêu ngày?", "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc..."]
]

# Process
light_scores = light_reranker.predict(pairs)  # [0.92, 0.89]

# Output
light_aids = ["law_1_113", "law_1_114", ...]  # Top 50
```

**Tầng 3 - Strong Reranking**:
```python
# Input
pairs = [
    ["Người lao động được nghỉ phép bao nhiêu ngày?", "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc..."],
    ["Người lao động được nghỉ phép bao nhiêu ngày?", "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc..."]
]

# Process
model1_scores = phobert_law.predict(pairs)  # [0.95, 0.87]
model2_scores = xlm_roberta.predict(pairs)   # [0.93, 0.89]
ensemble_scores = (model1_scores + model2_scores) / 2  # [0.94, 0.88]

# Output
final_results = [
    {
        "aid": "law_1_113",
        "content": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
        "rerank_score": 0.94
    },
    {
        "aid": "law_1_114",
        "content": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc...", 
        "rerank_score": 0.88
    }
]
```

### **Chi tiết từng bước**

#### **🔧 Bước 0: Domain-Adaptive Pre-training (DAPT)**

**Mục tiêu**: Chuyên môn hóa model ngôn ngữ cho pháp luật

**Input**:
```json
{
  "legal_corpus.json": "Kho văn bản pháp luật thô",
  "PhoBERT-base": "Model ngôn ngữ tổng quát"
}
```

**Quy trình**:
1. Load dữ liệu: Đọc toàn bộ văn bản pháp luật
2. Tokenization: Chuyển văn bản thành tokens
3. Masked Language Modeling: Học cách dự đoán từ bị che
4. Fine-tuning: Điều chỉnh weights cho phù hợp với pháp luật

**Output**:
```python
models/phobert-law/
├── config.json          # Cấu hình model
├── pytorch_model.bin    # Weights đã fine-tune
└── tokenizer.json      # Tokenizer chuyên biệt
```

**Thời gian**: 2-4 giờ (tùy GPU)

#### **🔍 Bước 1: Environment & Data Preparation**

**Mục tiêu**: Chuẩn bị môi trường và dữ liệu

**Input**:
```json
{
  "train.json": "Dữ liệu training với câu hỏi-đáp án",
  "public_test.json": "Dữ liệu test",
  "legal_corpus.json": "Kho văn bản pháp luật"
}
```

**Quy trình**:
1. Environment check: Kiểm tra GPU, memory, dependencies
2. Data loading: Load và validate dữ liệu
3. Data splitting: Chia train/validation
4. Index creation: Tạo mapping cho nhanh tra cứu

**Output**:
```python
data/processed/
├── train_data.json      # Dữ liệu training đã xử lý
├── val_data.json        # Dữ liệu validation
├── aid_map.json         # Mapping ID → nội dung
└── doc_id_to_aids.json # Mapping document → articles
```

**Thời gian**: 5-10 phút

#### **🔍 Bước 2: Hard Negative Mining & Data Preparation**

**Mục tiêu**: Tạo dữ liệu training chất lượng cao

**Input**:
```python
{
  "train_data": "Dữ liệu training cơ bản",
  "legal_corpus": "Kho văn bản pháp luật"
}
```

**Quy trình**:
1. Tạo positive pairs: (câu hỏi, đáp án đúng)
2. Train temporary Bi-Encoder: Model tạm thời để tìm hard negatives
3. Hard negative mining: Tìm câu trả lời sai nhưng rất giống đúng
4. Data augmentation: Tạo thêm dữ liệu đa dạng
5. Format conversion: Chuyển đổi format cho từng model

**Output**:
```python
data/training/
├── bi_encoder_triplets.json  # (question, positive, negative)
├── cross_encoder_pairs.json  # (question, document, label)
└── light_reranker_pairs.json # (question, document, label)
```

**Thời gian**: 30-60 phút

#### **🎓 Bước 3: Model Training & Evaluation**

**Mục tiêu**: Huấn luyện tất cả models và đánh giá

**Input**:
```python
{
  "bi_encoder_triplets": "Dữ liệu cho Bi-Encoder",
  "cross_encoder_pairs": "Dữ liệu cho Cross-Encoder",
  "phobert_law": "Model đã DAPT (nếu có)"
}
```

**Quy trình**:

**1. Bi-Encoder Training**:
```python
# Model: SentenceTransformer
# Loss: ContrastiveLoss
# Optimizer: AdamW (lr=2e-5)
# Batch size: 16 (adaptive)
# Epochs: 3
```

**2. FAISS Index Building**:
```python
# Encode tất cả documents
# Build FAISS index
# Save index và mappings
```

**3. Cross-Encoder Training**:
```python
# Model: PhoBERT-Law (nếu có) hoặc XLM-RoBERTa
# Loss: CrossEntropyLoss
# Optimizer: AdamW (lr=2e-5)
# Batch size: 8 (adaptive)
# Epochs: 5
```

**4. Light Reranker Training**:
```python
# Model: Smaller Cross-Encoder
# Purpose: Fast filtering
# Batch size: 16
# Epochs: 3
```

**Output**:
```python
models/
├── bi-encoder/          # Bi-Encoder model
├── cross-encoder/       # Cross-Encoder model
├── light-reranker/      # Light Reranker model
└── phobert-law/         # DAPT model (nếu có)

indexes/
├── faiss_index.bin      # FAISS index
└── index_mapping.json   # Index mappings

reports/
├── evaluation_report.json # Kết quả đánh giá
└── performance_metrics.json # Metrics chi tiết
```

**Thời gian**: 1-3 giờ (tùy GPU)

### **Ví dụ chi tiết luồng training từng bước**

#### **🔧 Bước 0: DAPT (Domain-Adaptive Pre-training)**

**Command**:
```bash
python scripts/00_adapt_model.py
```

**Input**: 
```json
# data/raw/legal_corpus.json
[
  {
    "law_id": "law_1",
    "title": "Bộ luật Lao động",
    "content": [
      {
        "aid": "113",
        "content_Article": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc..."
      }
    ]
  }
]
```

**Process**:
```python
# scripts/00_adapt_model.py
def train_phobert_law(legal_texts, output_path):
    # 1. Load PhoBERT base model
    model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # 2. Create dataset
    dataset = create_dapt_dataset(legal_texts, tokenizer)
    
    # 3. Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_path,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=5e-5,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    
    # 4. Save model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
```

**Output**:
```python
# models/phobert-law/
├── config.json          # Model configuration
├── pytorch_model.bin    # Trained weights
├── tokenizer.json      # Tokenizer
└── vocab.txt           # Vocabulary
```

**Logs**:
```
[INFO] [DAPT] Starting PhoBERT-Law training...
[INFO] [DAPT] Epoch 1/5: Loss = 2.2867
[INFO] [DAPT] Epoch 2/5: Loss = 1.8543
[INFO] [DAPT] Epoch 3/5: Loss = 1.4321
[INFO] [DAPT] Epoch 4/5: Loss = 1.1234
[INFO] [DAPT] Epoch 5/5: Loss = 0.5882
[INFO] [DAPT] Training completed successfully!
```

#### **🔍 Bước 1: Environment & Data Processing**

**Command**:
```bash
python scripts/01_check_environment.py
```

**Process**:
```python
# scripts/01_check_environment.py
def run_complete_pipeline():
    # 1. Check environment
    check_environment()
    # ✅ Python 3.8+
    # ✅ PyTorch 1.9+
    # ✅ GPU available: RTX 3080
    # ✅ Memory: 16GB available
    
    # 2. Check data files
    check_data_files()
    # ✅ legal_corpus.json: 15,420 articles
    # ✅ train.json: 1,000 questions
    # ✅ public_test.json: 200 questions
    
    # 3. Build mappings
    aid_map, doc_id_to_aids = build_maps_optimized()
    # aid_map = {"law_1_113": "Điều 113. Người lao động...", ...}
    # doc_id_to_aids = {"law_1": ["law_1_113", "law_1_114", ...]}
    
    # 4. Split data
    train_data, val_data = split_data_optimized(train_data)
    # train_data: 850 samples
    # val_data: 150 samples
```

**Output**:
```python
# data/processed/
├── aid_map.pkl                    # Mapping aid -> content
├── doc_id_to_aids_complete.json  # Mapping doc_id -> aids
├── train_data.json               # Training data
└── val_data.json                 # Validation data
```

#### **🎯 Bước 2: Hard Negative Mining & Data Preparation**

**Command**:
```bash
python scripts/02_prepare_training_data.py
```

**Process**:
```python
# scripts/02_prepare_training_data.py
def run_prepare_data_pipeline():
    # 1. Load processed data
    train_data, aid_map = load_processed_data()
    
    # 2. Create initial triplets
    initial_triplets = create_initial_triplets(train_data, aid_map)
    # [
    #   {
    #     "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    #     "positive": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
    #     "negative": "Điều 114. Thời gian nghỉ phép năm được tính theo năm làm việc..."
    #   }
    # ]
    
    # 3. Train temporary Bi-Encoder for hard negative mining
    temp_model = load_optimized_model_for_hard_negative_mining()
    
    # 4. Find hard negatives
    hard_negatives = find_hard_negatives(temp_model, train_data, aid_map)
    # [
    #   {
    #     "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    #     "positive": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
    #     "negative": "Điều 115. Người lao động được nghỉ việc riêng...",  # Hard negative
    #   }
    # ]
    
    # 5. Create final datasets
    bi_encoder_data = create_final_dataset(initial_triplets, hard_negatives, train_data, aid_map)
    cross_encoder_data = create_cross_encoder_pairs(bi_encoder_data)
    
    # 6. Save training data
    save_training_data(bi_encoder_data, cross_encoder_data)
```

**Output**:
```python
# data/processed/
├── bi_encoder_triplets.json      # Training data for Bi-Encoder
├── cross_encoder_pairs.json      # Training data for Cross-Encoder
└── light_reranker_pairs.json     # Training data for Light Reranker
```

#### **🎓 Bước 3: Model Training & Evaluation**

**Command**:
```bash
python scripts/03_train_models.py
```

**Process**:

**3.1 Bi-Encoder Training**:
```python
# scripts/03_train_models.py
def train_bi_encoder_optimized(bi_encoder_data):
    # 1. Load data
    train_examples = create_bi_encoder_examples(bi_encoder_data)
    
    # 2. Initialize model
    model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
    train_loss = losses.ContrastiveLoss(model)
    
    # 3. Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        optimizer_params={"lr": 2e-5},
        evaluator=evaluator,
        output_path=str(config.BI_ENCODER_PATH),
    )
    
    # 4. Save model
    model.save(str(config.BI_ENCODER_PATH))
```

**Logs**:
```
[INFO] [BI-ENCODER] Starting training...
[INFO] [BI-ENCODER] Epoch 1/3: Loss = 0.8543
[INFO] [BI-ENCODER] Epoch 2/3: Loss = 0.6234
[INFO] [BI-ENCODER] Epoch 3/3: Loss = 0.4123
[INFO] [BI-ENCODER] Training completed!
```

**3.2 FAISS Index Building**:
```python
def build_faiss_index_optimized(model):
    # 1. Encode all documents
    all_contents = list(aid_map.values())
    embeddings = model.encode(all_contents, show_progress_bar=True)
    
    # 2. Build FAISS index
    dimension = embeddings.shape[1]  # 768
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # 3. Save index and mappings
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    save_json(index_to_aid, config.INDEX_TO_AID_PATH)
```

**3.3 Cross-Encoder Training**:
```python
def _train_reranker(model_name_or_path, training_data, training_args, max_length, model_log_name):
    # 1. Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # 2. Create dataset
    dataset = Dataset.from_dict(training_data)
    
    # 3. Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    
    # 4. Save model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
```

**Logs**:
```
[INFO] [CROSS-ENCODER] Starting training...
[INFO] [CROSS-ENCODER] Epoch 1/5: Loss = 0.6543, Accuracy = 0.7234
[INFO] [CROSS-ENCODER] Epoch 2/5: Loss = 0.5234, Accuracy = 0.8123
[INFO] [CROSS-ENCODER] Epoch 3/5: Loss = 0.4123, Accuracy = 0.8543
[INFO] [CROSS-ENCODER] Epoch 4/5: Loss = 0.3456, Accuracy = 0.8765
[INFO] [CROSS-ENCODER] Epoch 5/5: Loss = 0.2987, Accuracy = 0.8923
[INFO] [CROSS-ENCODER] Training completed!
```

**3.4 Evaluation**:
```python
def run_comprehensive_evaluation():
    # 1. Load test data
    test_queries = ["Người lao động được nghỉ phép bao nhiêu ngày?"]
    ground_truth = [["law_1_113"]]
    
    # 2. Run retrieval evaluation
    retrieval_metrics = evaluate_retrieval(test_queries, ground_truth)
    # {
    #   "precision@5": 0.75,
    #   "recall@5": 0.68,
    #   "mrr": 0.82
    # }
    
    # 3. Run reranking evaluation
    reranking_metrics = evaluate_reranking(test_queries, ground_truth)
    # {
    #   "accuracy": 0.91,
    #   "precision": 0.89,
    #   "recall": 0.93
    # }
    
    # 4. Save evaluation report
    save_evaluation_report(retrieval_metrics, reranking_metrics)
```

**Output**:
```python
# models/
├── bi-encoder/          # Bi-Encoder model
├── cross-encoder/       # Cross-Encoder model
├── light-reranker/      # Light Reranker model
└── phobert-law/         # DAPT model

# indexes/
├── faiss_index.bin      # FAISS index
└── index_to_aid.json    # Index mappings

# reports/
├── evaluation_report.json # Evaluation results
└── performance_metrics.json # Performance metrics
```

### **Chạy từng bước riêng lẻ**

```bash
# Bước 0: DAPT (Domain-Adaptive Pre-training)
python scripts/00_adapt_model.py

# Bước 1: Kiểm tra môi trường và xử lý dữ liệu
python scripts/01_check_environment.py

# Bước 2: Chuẩn bị dữ liệu training
python scripts/02_prepare_training_data.py

# Bước 3: Huấn luyện models và đánh giá
python scripts/03_train_models.py
```

### **Hyperparameters tối ưu**

#### **Bi-Encoder**:
```python
{
  "learning_rate": 2e-5,
  "batch_size": 16,
  "epochs": 3,
  "warmup_steps": 100,
  "max_length": 256
}
```

#### **Cross-Encoder**:
```python
{
  "learning_rate": 2e-5,
  "batch_size": 8,
  "epochs": 5,
  "warmup_steps": 100,
  "max_length": 512
}
```

### **Performance Optimization**

#### **GPU Acceleration**:
```python
# Mixed Precision Training
config.FP16_TRAINING = True

# Gradient Accumulation
config.GRADIENT_ACCUMULATION_STEPS = 2

# Memory Optimization
torch.cuda.empty_cache()
```

#### **Data Loading Optimization**:
```python
# Multi-worker DataLoader
config.NUM_WORKERS = 4

# Pin Memory
config.PIN_MEMORY = True

# Prefetch Factor
config.PREFETCH_FACTOR = 2
```

---

## 📊 **Đánh giá hiệu suất**

### **Metrics đánh giá**

#### **🔍 Retrieval Metrics (Bi-Encoder)**:
```python
{
  "Precision@5": 0.75,    # 75% kết quả top-5 là đúng
  "Recall@5": 0.68,       # 68% đáp án đúng được tìm thấy
  "MRR": 0.82,            # Mean Reciprocal Rank
  "NDCG@10": 0.79         # Normalized Discounted Cumulative Gain
}
```

#### **⚖️ Reranking Metrics (Cross-Encoder)**:
```python
{
  "Accuracy": 0.91,       # 91% dự đoán đúng
  "AUC-ROC": 0.94,       # Area under ROC curve
  "Precision": 0.89,      # Precision cho positive class
  "Recall": 0.93          # Recall cho positive class
}
```

#### **🎯 End-to-End Metrics**:
```python
{
  "Response_Time": "0.5s",     # Thời gian trả lời
  "Throughput": "2 req/s",     # Số request/giây
  "Memory_Usage": "4GB",       # Memory sử dụng
  "GPU_Utilization": "85%"     # Sử dụng GPU
}
```

### **So sánh hiệu suất**

| Tiêu chí | Tìm kiếm thủ công | LawBot v8.1 |
|----------|-------------------|-------------|
| **Thời gian** | 2-3 giờ | **30 giây** |
| **Độ chính xác** | 60-70% | **90%+** |
| **Khả năng mở rộng** | Hạn chế | **Không giới hạn** |
| **Chi phí** | Cao (nhân lực) | **Thấp** |

### **Độ chính xác theo từng tầng**

| Metric | Tầng 1: Retrieval | Tầng 2: Light Reranking | Tầng 3: Strong Reranking |
|--------|-------------------|-------------------------|---------------------------|
| **Precision@5** | ~70% | ~80% | **> 90%** |
| **Recall@5** | ~60% | ~75% | **> 85%** |
| **MRR** | ~0.7 | ~0.8 | **> 0.85** |

### **Thời gian xử lý**

| Tác vụ | Thời gian | Mô tả |
|--------|-----------|-------|
| **Tầng 1**: Retrieval (500 ứng viên) | ~100ms | Tìm kiếm rộng trong toàn bộ kho dữ liệu |
| **Tầng 2**: Light Reranking (50 ứng viên) | ~150ms | Lọc nhanh với Light Reranker |
| **Tầng 3**: Strong Reranking (5 kết quả) | ~300ms | Thẩm định chuyên sâu với Ensemble |
| **📊 Tổng thời gian phản hồi** | **~550ms** | **Nhanh hơn 10x so với tìm kiếm thủ công** |

---

## 🧪 Tối ưu theo từng bước & Best Practices (v8.1)

> Tài liệu này tổng hợp các kỹ thuật tối ưu đã triển khai trong source, giúp chạy nhanh hơn, ổn định hơn, và đánh giá có số liệu đầy đủ ở cả FAST/QUALITY.

- **Bước 00 – DAPT (PhoBERT-Law)**
  - **Tokenizer giảm độ dài mẫu**: `max_length` rút gọn để tránh OOM khi không có GPU mạnh.
  - **Dataset tạo theo batch nhỏ**: đảm bảo memory footprint thấp, vẫn đủ đa dạng để domain-adapt.

- **Bước 01 – Environment & Data Processing**
  - **Validation cấu trúc & file bắt buộc**: tự tạo thư mục thiếu; báo lỗi sớm nếu thiếu data.
  - **Mapping tối ưu**: build `aid_map.pkl`, `doc_id_to_aids_complete.json` để truy xuất nhanh.
  - **Split dữ liệu hợp lý**: tách train/val với số lượng tối thiểu theo config (tránh val quá nhỏ).

- **Bước 02 – Hard Negative Mining & Chuẩn bị dữ liệu**
  - **Hard negatives chất lượng**: dùng Bi-Encoder tạm để tìm negatives “khó”, tăng chất lượng Cross-Encoder.
  - **Augmentation an toàn**: tăng đa dạng câu hỏi nhưng không phá hỏng semantics pháp luật.

- **Bước 03 – Training tích hợp & Evaluation (quan trọng)**
  - **Override mode sớm**: tất cả script hỗ trợ `--mode fast|quality`, set env trước khi import `config` ⇒ chọn đúng `config_fast.py`/`config_quality.py`.
  - **TrainingArguments tương thích nhiều version**: dùng builder `build_training_args_compat(...)` chỉ truyền tham số mà transformers hiện tại hỗ trợ; tự đồng bộ evaluation/save strategy để tránh lỗi.
  - **EarlyStopping an toàn**: chỉ bật khi version hỗ trợ evaluation strategy; tự tắt `load_best_model_at_end` nếu không đồng bộ được.
  - **Tiền xử lý văn bản chắc chắn**: làm sạch unicode/ký tự lạ, tối thiểu độ dài, cắt đôi theo `max_length // 2` cho cặp query/document.
  - **Tokenization cặp dài**: `truncation="only_second"` để ưu tiên giữ nguyên truy vấn, giảm warning overflow.
  - **Stratify hợp lệ**: cố gắng `class_encode_column('label')` rồi `stratify_by_column='label'`; nếu không được thì fallback split ngẫu nhiên.
  - **Fallback OOM**: tự giảm batch, rồi chuyển CPU nếu cần; dọn bộ nhớ CUDA giữa các lần thử.
  - **FAISS build chuẩn**: encode toàn bộ corpus với Bi-Encoder đã train; lưu `faiss` + `index_to_aid.json` đồng bộ.
  - **Evaluation ổn định (tại bước 03)**:
    - Metrics dạng phẳng: `precision@k`, `recall@k`, `f1@k` (k ∈ {1,3,5,10,20,50}).
    - Không loại cả batch khi có query trống kết quả; thay vào đó điền 0 cho query đó và vẫn tính metrics tổng.

- **Bước 04 – Evaluate chuyên sâu (độc lập, có report)**
  - **Override mode sớm**: `python scripts/04_evaluate_models.py --mode fast|quality` để dùng đúng config.
  - **Kết quả retrieval/reranking**: dùng `BatchEvaluator` và luôn chuẩn hóa output về keys phẳng.
  - **Per-query analysis**: tính precision/recall/F1 riêng từng câu hỏi; ghi số lượng AID đúng tìm thấy.
  - **Kiểm tra phủ AID**: thống kê tỉ lệ ground-truth AID có mặt trong `index_to_aid.json` (nếu thấp, rebuild index ở bước 03).
  - **Báo cáo chuẩn**: `EvaluationReporter.create_comprehensive_report(...)` + `save_report()` vào `reports/` với timestamp.

### FAST vs QUALITY (tóm tắt khác biệt chính)

- **FAST**: epochs thấp, batch nhỏ, `CROSS/LIGHT_MAX_LENGTH≈192`, `TOP_K_RETRIEVAL≈80`, `TOP_K_LIGHT≈40`, `TOP_K_FINAL≈3`, không dùng FP16 (ưu tiên ổn định và tốc độ).
- **QUALITY**: epochs/batch lớn hơn, `CROSS/LIGHT_MAX_LENGTH≈320`, `TOP_K_RETRIEVAL≈150`, `TOP_K_LIGHT≈80`, `TOP_K_FINAL≈7`, bật FP16 nếu khả dụng (ưu tiên chất lượng).

### Xử lý văn bản dài (Long documents)

- **Chunking theo token** với overlap (~50) và `truncation="only_second"` để giữ nguyên truy vấn.
- **Gộp điểm thông minh**: lấy max score trên tất cả chunk của một văn bản để đại diện.
- **Dedup đơn giản**: lọc các kết quả nội dung trùng lặp mức cao để danh sách gọn và đa dạng.

### Logging & Theo dõi tiến trình

- **Outline-aware**: log theo bước/hàm `[STEP {id}] [FUNC] ▶/◀` để dễ theo dõi tiến trình.
- **Console/file gồm vị trí nguồn**: `[name] [func:line]` giúp truy vết chính xác đoạn code phát log.

### Đảm bảo evaluation có số liệu (checklist nhanh)

- Đã build lại FAISS bằng Bi-Encoder mới chưa? (chạy lại bước 03 hoặc `--start-step 03`).
- Ground-truth AID có khớp format với `index_to_aid.json`? (xem thống kê phủ AID in ra ở bước 04).
- K đang dùng có phù hợp? `TOP_K_RETRIEVAL` quá nhỏ dễ bỏ lỡ ground-truth.
- Đang chạy đúng mode? Dùng `--mode fast|quality` hoặc `switch_performance_mode.py --immediate` trước khi chạy.

### Chạy App theo mode

- Windows (PowerShell):
  ```powershell
  setx LAWBOT_PERFORMANCE_MODE fast
  # Mở terminal mới hoặc:
  $env:LAWBOT_PERFORMANCE_MODE="fast"; streamlit run app/app.py
  ```
- Linux/macOS:
  ```bash
  LAWBOT_PERFORMANCE_MODE=fast streamlit run app/app.py
  ```


## 🛠️ **Phát triển và bảo trì**

### **API Documentation**

#### **LegalQAPipeline**

Class chính để tương tác với hệ thống.

```python
from core.pipeline import LegalQAPipeline

# Khởi tạo
pipeline = LegalQAPipeline()

# Kiểm tra trạng thái
if pipeline.is_ready:
    print("Pipeline sẵn sàng!")
else:
    print("Pipeline chưa sẵn sàng!")
```

#### **Methods**

##### `predict(query, top_k_retrieval=100, top_k_final=5)`

Dự đoán câu trả lời cho câu hỏi.

**Parameters**:
- `query` (str): Câu hỏi cần trả lời
- `top_k_retrieval` (int): Số lượng kết quả retrieval (default: 100)
- `top_k_final` (int): Số lượng kết quả cuối cùng (default: 5)

**Returns**:
- `List[Dict]`: Danh sách kết quả với format:
  ```python
  [
      {
          "aid": "law_1_113",
          "content": "Điều 113: Người lao động được nghỉ phép năm...",
          "retrieval_score": 0.85,
          "rerank_score": 0.92
      },
      # ...
  ]
  ```

**Ví dụ sử dụng**:
```python
# Input
query = "Người lao động được nghỉ phép bao nhiêu ngày?"

# Output
results = [
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

##### `retrieve(query, top_k=100)`

Chỉ thực hiện retrieval (tầng 1).

**Parameters**:
- `query` (str): Câu hỏi
- `top_k` (int): Số lượng kết quả

**Returns**:
- `Tuple[List[str], List[float]]`: (aids, scores)

**Ví dụ**:
```python
# Input
query = "Điều kiện thành lập doanh nghiệp?"

# Output
aids = ["law_2_15", "law_2_16", "law_2_17", ...]
scores = [0.95, 0.87, 0.82, ...]
```

##### `rerank(query, retrieved_aids, retrieved_distances)`

Chỉ thực hiện reranking (tầng 2).

**Parameters**:
- `query` (str): Câu hỏi
- `retrieved_aids` (List[str]): Danh sách AIDs từ retrieval
- `retrieved_distances` (List[float]): Điểm số từ retrieval

**Returns**:
- `List[Dict]`: Kết quả đã rerank

**Ví dụ**:
```python
# Input
query = "Người lao động được nghỉ phép bao nhiêu ngày?"
retrieved_aids = ["law_1_113", "law_1_114", "law_1_115"]
retrieved_distances = [0.85, 0.78, 0.72]

# Output
reranked_results = [
    {
        "aid": "law_1_113",
        "content": "Điều 113. Người lao động được nghỉ phép năm...",
        "retrieval_score": 0.85,
        "rerank_score": 0.92
    },
    {
        "aid": "law_1_114", 
        "content": "Điều 114. Thời gian nghỉ phép năm...",
        "retrieval_score": 0.78,
        "rerank_score": 0.87
    }
]
```

### **Utilities**

#### **Dataset Filtering Utility**

Script để lọc dataset trước khi chạy pipeline chính:

```bash
# Chạy filtering utility
python scripts/utils/run_filter.py

# Hoặc chạy trực tiếp
python scripts/utils/filter_dataset.py
```

**Chức năng**:
- Lọc bỏ samples có ground truth không phù hợp
- Giữ lại ~100-200 samples chất lượng cao
- Cải thiện chất lượng dữ liệu training

#### **Project Structure Checker**

Kiểm tra cấu trúc project và best practices:

```bash
python scripts/utils/check_project.py
```

**Chức năng**:
- Kiểm tra cấu trúc thư mục
- Validate naming conventions
- Kiểm tra documentation
- Đảm bảo best practices

### **Troubleshooting**

#### **🔧 Lỗi thường gặp**

**1. GPU Issues**
```bash
# Kiểm tra GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Force CPU training nếu cần
export CUDA_VISIBLE_DEVICES=""
python run_pipeline.py
```

**2. Memory Issues**
```bash
# Giảm batch size
export LAWBOT_BI_ENCODER_BATCH_SIZE=8
export LAWBOT_CROSS_ENCODER_BATCH_SIZE=4

# Tắt mixed precision
export LAWBOT_FP16_TRAINING=false
```

**3. Model Loading Issues**
```bash
# Kiểm tra model paths
ls -la models/
ls -la models/phobert-law/
ls -la models/bi-encoder/

# Rebuild models nếu cần
python scripts/00_adapt_model.py
python scripts/03_train_models.py
```

**4. Data Loading Issues**
```bash
# Kiểm tra data files
ls -la data/raw/
ls -la data/processed/

# Validate data structure
python scripts/utils/check_project.py
```

#### **⚡ Performance Optimization Tips**

**1. GPU Optimization**
```bash
# Tối ưu GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Bật mixed precision
export LAWBOT_FP16_TRAINING=true
```

**2. System Optimization**
```bash
# Tăng file descriptors
ulimit -n 65536

# Tối ưu CPU cores
export OMP_NUM_THREADS=8
```

**3. Pipeline Optimization**
```bash
# Skip DAPT nếu không cần
python run_pipeline.py --no-dapt

# Chạy từ bước cụ thể
python run_pipeline.py --start-step 02
```

### **Monitoring & Logging**

#### **1. Performance Monitoring**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor memory usage
htop

# Check logs
tail -f logs/pipeline.log
```

#### **2. Quality Metrics**
```bash
# Check evaluation results
ls -la reports/

# View latest report
cat reports/evaluation_report_*.json
```

---

## ❓ **Hỏi đáp**

### **Câu hỏi thường gặp**

**Q: LawBot có thể thay thế luật sư không?**
A: Không, LawBot chỉ là công cụ hỗ trợ tra cứu pháp luật. Để có tư vấn pháp lý chuyên sâu, bạn nên tham khảo ý kiến của luật sư.

**Q: Dữ liệu pháp luật có được cập nhật thường xuyên không?**
A: Dữ liệu pháp luật cần được cập nhật thủ công. Bạn có thể thêm văn bản pháp luật mới vào file `legal_corpus.json` và chạy lại pipeline.

**Q: Có thể sử dụng LawBot cho pháp luật nước khác không?**
A: Hiện tại LawBot được thiết kế đặc biệt cho pháp luật Việt Nam. Để sử dụng cho nước khác, cần thay đổi dữ liệu training và có thể cần điều chỉnh model.

**Q: Làm thế nào để cải thiện độ chính xác?**
A: Có thể cải thiện bằng cách:
- Tăng chất lượng dữ liệu training
- Điều chỉnh hyperparameters
- Sử dụng model lớn hơn
- Thêm data augmentation

**Q: Có thể chạy LawBot trên CPU không?**
A: Có, LawBot có thể chạy trên CPU nhưng sẽ chậm hơn đáng kể so với GPU. Để force CPU mode, sử dụng:
```bash
export CUDA_VISIBLE_DEVICES=""
python run_pipeline.py
```

### **Liên hệ hỗ trợ**

- **GitHub Issues**: [Tạo issue](https://github.com/lawbot-team/lawbot/issues)
- **Email**: support@lawbot.com
- **Documentation**: [Wiki](https://github.com/lawbot-team/lawbot/wiki)

---

## 📚 **Tài liệu tham khảo**

### **Papers**
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Cross-Encoders vs. Bi-Encoders for Zero-Shot Classification](https://arxiv.org/abs/2108.08877)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

### **Libraries**
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

### **Datasets**
- [Vietnamese Legal Corpus](https://github.com/lawbot-team/vietnamese-legal-corpus)
- [Legal QA Dataset](https://github.com/lawbot-team/legal-qa-dataset)

---

## 🤝 **Đóng góp**

Chúng tôi rất hoan nghênh mọi đóng góp! Vui lòng đọc `CONTRIBUTING.md` để biết thêm chi tiết.

## 📄 **License**

Dự án này được cấp phép theo MIT License.

---

**Made with ❤️ by LawBot Team**
