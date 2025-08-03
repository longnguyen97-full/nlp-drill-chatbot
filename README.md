# 🏛️ LawBot - Legal QA Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/lawbot-team/lawbot)

> **Hệ thống hỏi-đáp pháp luật thông minh cho Việt Nam**  
> Sử dụng kiến trúc Retrieval-Rerank với Bi-Encoder và Cross-Encoder

## 📋 **TỔNG QUAN (WHAT)**

### **LawBot là gì?**
LawBot là một hệ thống hỏi-đáp pháp luật tiên tiến được thiết kế đặc biệt cho pháp luật Việt Nam. Hệ thống sử dụng công nghệ AI hiện đại để trả lời các câu hỏi về pháp luật một cách chính xác và nhanh chóng.

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

### **Cách hoạt động:**
1. **🔍 Retrieval (Bi-Encoder)**: Tìm 100 văn bản pháp luật liên quan nhất
2. **⚖️ Reranking (Cross-Encoder)**: Đánh giá và sắp xếp lại top 5 kết quả chính xác nhất
3. **📤 Output**: Trả về 5 văn bản pháp luật phù hợp nhất với điểm số

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

## 🚀 **QUICK START**

### **Bước 1: Cài đặt**

```bash
# Clone repository
git clone https://github.com/lawbot-team/lawbot.git
cd lawbot

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài đặt như package
pip install -e .
```

### **Bước 2: Kiểm tra môi trường**

```bash
python scripts/01_check_environment.py
```

### **Bước 3: Chạy pipeline**

```bash
# Chạy toàn bộ pipeline
python run_pipeline.py

# Chạy từ bước cụ thể
python run_pipeline.py --step 05

# Bỏ qua filtering
python run_pipeline.py --skip-filtering

# Xem danh sách bước
python run_pipeline.py --list-steps
```

### **Bước 4: Sử dụng API**

```python
from core.pipeline import LegalQAPipeline

# Khởi tạo pipeline
pipeline = LegalQAPipeline()

# Hỏi đáp
query = "Người lao động được nghỉ phép bao nhiêu ngày?"
results = pipeline.predict(
    query=query,
    top_k_retrieval=100,
    top_k_final=5
)

# In kết quả
for i, result in enumerate(results):
    print(f"Kết quả {i+1}: {result['aid']}")
    print(f"Điểm: {result['rerank_score']:.4f}")
    print(f"Nội dung: {result['content'][:200]}...")
    print("-" * 50)
```

## 📁 **CẤU TRÚC PROJECT**

```
LawBot/
├── 📁 core/                    # Core utilities
│   ├── logging_utils.py        # Unified logging
│   ├── data_utils.py           # Data processing
│   ├── model_utils.py          # Model utilities
│   ├── evaluation_utils.py     # Evaluation metrics
│   └── pipeline_utils.py       # Pipeline orchestration
│
├── 📁 scripts/                 # Pipeline scripts
│   ├── 01_check_environment.py
│   ├── 02_filter_dataset.py
│   ├── 03_preprocess_data.py
│   ├── 04_split_data.py
│   ├── 05_validate_mapping.py
│   ├── 06_prepare_training_data.py
│   ├── 07_merge_data.py
│   ├── 08_augment_data.py
│   ├── 09_train_bi_encoder.py
│   ├── 10_build_faiss_index.py
│   ├── 11_train_cross_encoder.py
│   └── 12_evaluate_pipeline.py
│
├── 📁 data/                    # Data directories
│   ├── raw/                    # Original data
│   ├── processed/              # Processed data
│   └── validation/             # Validation data
│
├── 📁 models/                  # Trained models
│   ├── bi_encoder/
│   └── cross_encoder/
│
├── 📁 indexes/                 # FAISS indexes
├── 📁 reports/                 # Evaluation reports
├── 📁 logs/                    # Log files
├── 📁 app/                     # Web application
│
├── 📄 config.py                # Configuration
├── 📄 run_pipeline.py          # Main pipeline runner
├── 📄 setup.py                 # Package setup
├── 📄 requirements.txt         # Dependencies
└── 📄 README.md               # Documentation
```

## 🔧 **CẤU HÌNH**

### **Environment Variables**

Bạn có thể cấu hình LawBot thông qua environment variables:

```bash
# Environment
export LAWBOT_ENV=production
export LAWBOT_DEBUG=false

# Directories
export LAWBOT_DATA_DIR=/path/to/data
export LAWBOT_MODELS_DIR=/path/to/models
export LAWBOT_INDEXES_DIR=/path/to/indexes

# Hyperparameters
export LAWBOT_BI_ENCODER_BATCH_SIZE=8
export LAWBOT_CROSS_ENCODER_BATCH_SIZE=4
export LAWBOT_BI_ENCODER_LR=1e-7
export LAWBOT_CROSS_ENCODER_LR=5e-6

# Pipeline settings
export LAWBOT_TOP_K_RETRIEVAL=100
export LAWBOT_TOP_K_FINAL=5
```

### **Configuration File**

Tất cả cấu hình được quản lý trong `config.py`:

```python
import config

# In thông tin cấu hình
config.print_config_summary()

# Validate cấu hình
config.validate_config()
```

## 📊 **PIPELINE FLOW CHI TIẾT**

### **🎯 Quy trình 12 bước với Input/Output:**

#### **Bước 01: Kiểm tra môi trường**
```bash
# Input: Không có
python scripts/01_check_environment.py

# Output: Báo cáo môi trường
✅ Python version: 3.8.0
✅ CUDA available: True
✅ Required packages installed
✅ Data files found
✅ Directories created
```

#### **Bước 02: Lọc dataset chất lượng**
```bash
# Input: data/raw/train.json
python scripts/02_filter_dataset.py

# Output: data/raw/train_filtered.json
# Loại bỏ 70-90% samples có ground truth không phù hợp
# Giữ lại ~100-200 samples chất lượng cao
```

#### **Bước 03: Tiền xử lý dữ liệu (Fixed Mapping)**
```bash
# Input: 
# - data/raw/legal_corpus.json
# - data/raw/train_filtered.json

python scripts/03_preprocess_data.py

# Output:
# - data/processed/aid_map.pkl (Article ID mapping)
# - data/processed/doc_id_to_aids_complete.json (Document to AIDs mapping)
# - data/processed/train_fixed.json (Fixed training data)
```

**Ví dụ Input/Output:**
```json
// Input: legal_corpus.json
{
  "doc_id": 1,
  "content": [
    {
      "aid": "law_1_113",
      "content_Article": "Điều 113. Người lao động được nghỉ phép năm..."
    }
  ]
}

// Output: aid_map.pkl
{
  "law_1_113": "Điều 113. Người lao động được nghỉ phép năm...",
  "law_1_114": "Điều 114. Thời gian nghỉ phép năm được tính..."
}
```

#### **Bước 04: Chia dữ liệu train/validation**
```bash
# Input: data/processed/train_fixed.json
python scripts/04_split_data.py

# Output:
# - data/raw/train_split.json (85% training)
# - data/raw/validation_split.json (15% validation)
```

**Ví dụ Input/Output:**
```json
// Input: train_fixed.json (100 samples)
[
  {
    "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
    "relevant_aids": ["law_1_113", "law_1_114"]
  }
]

// Output: train_split.json (85 samples)
// Output: validation_split.json (15 samples)
```

#### **Bước 05: Validate mapping**
```bash
# Input: 
# - data/processed/aid_map.pkl
# - data/raw/validation_split.json

python scripts/05_validate_mapping.py

# Output: Báo cáo validation
✅ Mapping validation passed
✅ All AIDs in validation set exist in aid_map
✅ Ground truth format correct
```

#### **Bước 06: Chuẩn bị training data**
```bash
# Input: data/raw/train_split.json
python scripts/06_prepare_training_data.py

# Output:
# - data/processed/train_triplets_easy.jsonl (Bi-Encoder triplets)
# - data/processed/train_pairs.jsonl (Cross-Encoder pairs)
# - data/processed/bi_encoder_validation.jsonl (Validation data)
```

**Ví dụ Input/Output:**
```json
// Input: train_split.json
{
  "question": "Người lao động được nghỉ phép bao nhiêu ngày?",
  "relevant_aids": ["law_1_113", "law_1_114"]
}

// Output: train_triplets_easy.jsonl
{
  "anchor": "Người lao động được nghỉ phép bao nhiêu ngày?",
  "positive": "Điều 113. Người lao động được nghỉ phép năm...",
  "negative": "Điều 200. Quy định về thời gian làm việc..."
}

// Output: train_pairs.jsonl
{
  "texts": [
    "Người lao động được nghỉ phép bao nhiêu ngày?",
    "Điều 113. Người lao động được nghỉ phép năm..."
  ],
  "labels": 1
}
```

#### **Bước 07: Merge dữ liệu**
```bash
# Input: 
# - data/processed/train_triplets_easy.jsonl
# - data/processed/train_pairs.jsonl

python scripts/07_merge_data.py

# Output:
# - data/processed/bi_encoder_train_mixed.jsonl (Easy + Hard negatives)
# - data/processed/train_pairs_mixed.jsonl (Easy + Hard negatives)
```

#### **Bước 08: Augment dữ liệu**
```bash
# Input: 
# - data/processed/bi_encoder_train_mixed.jsonl
# - data/processed/train_pairs_mixed.jsonl

python scripts/08_augment_data.py

# Output:
# - data/processed/bi_encoder_train_augmented.jsonl (1.5x size)
# - data/processed/train_pairs_augmented.jsonl (1.3x size)
```

**Ví dụ Augmentation:**
```json
// Input
{
  "anchor": "Người lao động được nghỉ phép bao nhiêu ngày?",
  "positive": "Điều 113. Người lao động được nghỉ phép năm...",
  "negative": "Điều 200. Quy định về thời gian làm việc..."
}

// Output (Augmented)
{
  "anchor": "Người lao động được nghỉ phép bao nhiêu ngày?",
  "positive": "Điều 113. Người lao động được nghỉ phép năm 12 ngày làm việc...",
  "negative": "Điều 200. Quy định về thời gian làm việc và nghỉ ngơi..."
}
```

#### **Bước 09: Train Bi-Encoder**
```bash
# Input: data/processed/bi_encoder_train_augmented.jsonl
python scripts/09_train_bi_encoder.py

# Output: models/bi-encoder/
# - config.json
# - pytorch_model.bin
# - tokenizer.json
# - special_tokens_map.json
```

**Training Configuration:**
```python
# Hyperparameters
batch_size = 4
epochs = 1
learning_rate = 1e-7
warmup_steps = 50
eval_steps = 25

# Loss: MultipleNegativesRankingLoss
# Model: bkai-foundation-models/vietnamese-bi-encoder
```

#### **Bước 10: Build FAISS index**
```bash
# Input: 
# - models/bi-encoder/ (trained model)
# - data/processed/aid_map.pkl

python scripts/10_build_faiss_index.py

# Output:
# - indexes/legal.faiss (FAISS index)
# - indexes/index_to_aid.json (Index to AID mapping)
```

**Index Structure:**
```python
# FAISS Index
index_type = "IndexFlatIP"  # Inner Product
dimension = 768
normalize = True

# Index size: ~100MB for 100K documents
# Search time: 50-100ms for 100 results
```

#### **Bước 11: Train Cross-Encoder**
```bash
# Input: data/processed/train_pairs_augmented.jsonl
python scripts/11_train_cross_encoder.py

# Output: models/cross-encoder-v2/
# - config.json
# - pytorch_model.bin
# - tokenizer.json
```

**Training Configuration:**
```python
# Hyperparameters
batch_size = 4
epochs = 1
learning_rate = 5e-6
max_length = 256
gradient_accumulation_steps = 4

# Model: vinai/phobert-large
# Input format: [CLS] query [SEP] document [SEP]
# Output: Binary classification (relevant/not relevant)
```

#### **Bước 12: Đánh giá pipeline**
```bash
# Input: 
# - models/bi-encoder/
# - models/cross-encoder-v2/
# - indexes/legal.faiss
# - data/raw/validation_split.json

python scripts/12_evaluate_pipeline.py

# Output: reports/evaluation_report_*.json
```

**Evaluation Metrics:**
```json
{
  "retrieval_metrics": {
    "precision@1": 0.75,
    "precision@5": 0.80,
    "recall@5": 0.70,
    "f1@5": 0.75,
    "mrr": 0.78
  },
  "reranking_metrics": {
    "precision@1": 0.85,
    "precision@5": 0.82,
    "recall@5": 0.75,
    "f1@5": 0.78,
    "mrr": 0.82
  }
}
```

### **🧠 Kiến trúc xử lý chi tiết:**

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

## 📈 **PERFORMANCE**

### **🎯 Metrics dự kiến:**

| Metric | Bi-Encoder | Cross-Encoder | Pipeline |
|--------|------------|---------------|----------|
| **Precision@5** | 60-70% | 75-85% | 80-90% |
| **Recall@5** | 50-60% | 65-75% | 70-80% |
| **F1@5** | 55-65% | 70-80% | 75-85% |
| **MRR** | 0.6-0.7 | 0.7-0.8 | 0.75-0.85 |

### **⚡ Thời gian xử lý:**

| Component | Thời gian |
|-----------|-----------|
| **Bi-Encoder Retrieval** | 50-100ms |
| **Cross-Encoder Reranking** | 200-500ms |
| **Total Pipeline** | 250-600ms |

## 🛠️ **DEVELOPMENT**

### **Cài đặt development dependencies:**

```bash
pip install -e .[dev]
```

### **Chạy tests:**

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/
```

### **Code formatting:**

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📚 **API DOCUMENTATION**

### **LegalQAPipeline**

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

#### **Methods:**

##### `predict(query, top_k_retrieval=100, top_k_final=5)`

Dự đoán câu trả lời cho câu hỏi.

**Parameters:**
- `query` (str): Câu hỏi cần trả lời
- `top_k_retrieval` (int): Số lượng kết quả retrieval (default: 100)
- `top_k_final` (int): Số lượng kết quả cuối cùng (default: 5)

**Returns:**
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

**Ví dụ sử dụng:**
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

**Ví dụ sử dụng:**
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

**Parameters:**
- `query` (str): Câu hỏi
- `top_k` (int): Số lượng kết quả

**Returns:**
- `Tuple[List[str], List[float]]`: (aids, scores)

**Ví dụ:**
```python
# Input
query = "Điều kiện thành lập doanh nghiệp?"

# Output
aids = ["law_2_15", "law_2_16", "law_2_17", ...]
scores = [0.95, 0.87, 0.82, ...]
```

**Ví dụ:**
```python
# Input
query = "Điều kiện thành lập doanh nghiệp?"

# Output
aids = ["law_2_15", "law_2_16", "law_2_17", ...]
scores = [0.95, 0.87, 0.82, ...]
```

##### `rerank(query, retrieved_aids, retrieved_distances)`

Chỉ thực hiện reranking (tầng 2).

**Parameters:**
- `query` (str): Câu hỏi
- `retrieved_aids` (List[str]): Danh sách AIDs từ retrieval
- `retrieved_distances` (List[float]): Điểm số từ retrieval

**Returns:**
- `List[Dict]`: Kết quả đã rerank

**Ví dụ:**
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

**Ví dụ:**
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

## 🚨 **TROUBLESHOOTING**

### **Lỗi thường gặp:**

#### **1. "ModuleNotFoundError: No module named 'config'"**

**Nguyên nhân:** Python path không đúng.

**Giải pháp:**
```bash
# Thêm project root vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/lawbot"

# Hoặc chạy từ project root
cd /path/to/lawbot
python scripts/01_check_environment.py
```

#### **2. "CUDA out of memory"**

**Nguyên nhân:** GPU memory không đủ.

**Giải pháp:**
```bash
# Giảm batch size
export LAWBOT_BI_ENCODER_BATCH_SIZE=2
export LAWBOT_CROSS_ENCODER_BATCH_SIZE=1

# Hoặc sử dụng CPU
export CUDA_VISIBLE_DEVICES=""
```

#### **3. "File not found: data/raw/legal_corpus.json"**

**Nguyên nhân:** File dữ liệu chưa được tải.

**Giải pháp:**
```bash
# Kiểm tra cấu trúc thư mục
ls -la data/raw/

# Tải dữ liệu nếu cần
python scripts/download_data.py
```

#### **4. "Model did not return a loss"**

**Nguyên nhân:** Cross-Encoder training configuration sai.

**Giải pháp:**
```bash
# Kiểm tra labels trong training data
python scripts/debug_training_data.py

# Chạy lại training với config đúng
python scripts/11_train_cross_encoder.py
```

### **Debug mode:**

```bash
# Bật debug mode
export LAWBOT_DEBUG=true

# Chạy với logging chi tiết
python run_pipeline.py --step 01
```

## 📖 **TÀI LIỆU THAM KHẢO**

### **Papers:**
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Cross-Encoders vs. Bi-Encoders for Zero-Shot Classification](https://arxiv.org/abs/2108.08877)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

### **Libraries:**
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

### **Datasets:**
- [Vietnamese Legal Corpus](https://github.com/lawbot-team/vietnamese-legal-corpus)
- [Legal QA Dataset](https://github.com/lawbot-team/legal-qa-dataset)

## 🤝 **CONTRIBUTING**

Chúng tôi rất hoan nghênh mọi đóng góp! Vui lòng đọc [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết.

### **Cách đóng góp:**

1. **Fork** repository
2. **Tạo** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Tạo** Pull Request

### **Guidelines:**

- ✅ Tuân thủ PEP 8 style guide
- ✅ Viết tests cho tính năng mới
- ✅ Cập nhật documentation
- ✅ Kiểm tra code với linter

## 📄 **LICENSE**

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 📞 **LIÊN HỆ**

- **Email:** lawbot@example.com
- **GitHub:** [@lawbot-team](https://github.com/lawbot-team)
- **Documentation:** [https://lawbot.readthedocs.io/](https://lawbot.readthedocs.io/)
- **Issues:** [https://github.com/lawbot-team/lawbot/issues](https://github.com/lawbot-team/lawbot/issues)

## 🙏 **ACKNOWLEDGMENTS**

- **BKAI Foundation** cho Vietnamese Bi-Encoder model
- **VINAI** cho PhoBERT model
- **Facebook Research** cho FAISS library
- **Hugging Face** cho Transformers library

---

**Made with ❤️ by LawBot Team**
