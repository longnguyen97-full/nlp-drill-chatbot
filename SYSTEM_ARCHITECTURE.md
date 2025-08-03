# 🏗️ SYSTEM ARCHITECTURE - LawBot

> **Hướng dẫn chi tiết về kiến trúc hệ thống**  
> Giải thích từng component, luồng dữ liệu và tương tác

## 📋 **TỔNG QUAN KIẾN TRÚC**

### **Kiến trúc tổng thể:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Web UI    │  │   API REST  │  │  CLI Tools  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Pipeline  │  │   Cache     │  │  Monitoring │          │
│  │  Manager    │  │  Manager    │  │   System    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CORE AI LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Bi-Encoder  │  │Cross-Encoder│  │   FAISS     │          │
│  │ (Retrieval) │  │(Reranking)  │  │   Index     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Raw Data  │  │ Processed   │  │   Models    │          │
│  │   Storage   │  │   Data      │  │   Storage   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## 🔍 **RETRIEVAL LAYER (BI-ENCODER)**

### **Chức năng:**
- **Input**: Câu hỏi của người dùng
- **Output**: 100 văn bản pháp luật liên quan nhất
- **Thời gian**: 50-100ms

### **Kiến trúc chi tiết:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Query   │───▶│  Bi-Encoder     │───▶│  Query Vector   │
│   (Text)        │    │  Model          │    │  (768 dim)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Retrieved      │◀───│  FAISS Index    │◀───│  Similarity     │
│  Documents      │    │  Search         │    │  Search         │
│  (100 docs)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Components:**

#### **1. Bi-Encoder Model:**
```python
# Model: bkai-foundation-models/vietnamese-bi-encoder
# Architecture: Sentence Transformer
# Output dimension: 768
# Training: Multiple Negatives Ranking Loss
```

#### **2. FAISS Index:**
```python
# Index type: IndexFlatIP (Inner Product)
# Normalization: L2 normalization
# Search method: Exact search
# Index size: ~100MB for 100K documents
```

#### **3. Document Embeddings:**
```python
# Pre-computed embeddings for all legal documents
# Storage: FAISS index + metadata mapping
# Update frequency: Batch updates
```

### **Luồng xử lý:**
1. **Tokenization**: Chuyển câu hỏi thành tokens
2. **Encoding**: Tạo embedding vector (768 dim)
3. **Normalization**: L2 normalize vector
4. **Search**: Tìm kiếm trong FAISS index
5. **Ranking**: Sắp xếp theo similarity score
6. **Output**: Trả về top 100 documents

## ⚖️ **RERANKING LAYER (CROSS-ENCODER)**

### **Chức năng:**
- **Input**: Câu hỏi + 100 documents từ retrieval
- **Output**: Top 5 documents với điểm số chính xác
- **Thời gian**: 200-500ms

### **Kiến trúc chi tiết:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Query + Docs   │───▶│ Cross-Encoder   │───▶│  Relevance      │
│  (100 pairs)    │    │  Model          │    │  Scores         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Final Results  │◀───│  Score Ranking  │◀───│  Softmax        │
│  (Top 5)        │    │  & Sorting      │    │  Probabilities  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Components:**

#### **1. Cross-Encoder Model:**
```python
# Model: vinai/phobert-large
# Architecture: RoBERTa-based
# Input: [CLS] query [SEP] document [SEP]
# Output: Binary classification (relevant/not relevant)
```

#### **2. Text Processing:**
```python
# Chunking: Split long documents into chunks
# Overlap: 50 tokens between chunks
# Max length: 256 tokens per chunk
# Strategy: Take max score across chunks
```

#### **3. Batch Processing:**
```python
# Batch size: 4 (configurable)
# Memory optimization: Gradient accumulation
# Device: GPU with mixed precision (FP16)
```

### **Luồng xử lý:**
1. **Pair Creation**: Tạo cặp (query, document)
2. **Chunking**: Chia documents dài thành chunks
3. **Tokenization**: Tokenize từng cặp
4. **Batch Processing**: Xử lý theo batch
5. **Scoring**: Tính relevance score
6. **Aggregation**: Lấy max score cho mỗi document
7. **Ranking**: Sắp xếp theo score
8. **Output**: Trả về top 5

## 📊 **DATA FLOW**

### **Training Data Flow:**
```
Raw Data (legal_corpus.json, train.json)
    │
    ▼
Preprocessing (03_preprocess_data.py)
    │
    ▼
Data Splitting (04_split_data.py)
    │
    ▼
Training Data Preparation (06_prepare_training_data.py)
    │
    ▼
Data Augmentation (08_augment_data.py)
    │
    ▼
Model Training (09_train_bi_encoder.py, 11_train_cross_encoder.py)
```

### **Inference Data Flow:**
```
User Query
    │
    ▼
Bi-Encoder Retrieval
    │
    ▼
Top 100 Documents
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Top 5 Results
    │
    ▼
Formatted Response
```

## 🔧 **CONFIGURATION SYSTEM**

### **Environment Variables:**
```bash
# Core settings
LAWBOT_ENV=production
LAWBOT_DEBUG=false

# Paths
LAWBOT_DATA_DIR=/opt/lawbot/data
LAWBOT_MODELS_DIR=/opt/lawbot/models
LAWBOT_INDEXES_DIR=/opt/lawbot/indexes

# Performance
LAWBOT_BI_ENCODER_BATCH_SIZE=8
LAWBOT_CROSS_ENCODER_BATCH_SIZE=4
LAWBOT_TOP_K_RETRIEVAL=100
LAWBOT_TOP_K_FINAL=5

# GPU settings
LAWBOT_FP16_TRAINING=true
CUDA_VISIBLE_DEVICES=0
```

### **Configuration Validation:**
```python
# Auto-validation on import
config.validate_config()
config.print_config_summary()
```

## 📁 **FILE STRUCTURE**

### **Core Components:**
```
core/
├── logging_utils.py      # Unified logging system
├── data_utils.py         # Data processing utilities
├── model_utils.py        # Model loading and inference
├── evaluation_utils.py   # Performance metrics
└── pipeline_utils.py     # Pipeline orchestration
```

### **Pipeline Scripts:**
```
scripts/
├── 01_check_environment.py      # Environment validation
├── 02_filter_dataset.py         # Data quality filtering
├── 03_preprocess_data.py        # Data preprocessing
├── 04_split_data.py            # Train/validation split
├── 05_validate_mapping.py      # Mapping validation
├── 06_prepare_training_data.py # Training data preparation
├── 07_merge_data.py           # Data merging
├── 08_augment_data.py         # Data augmentation
├── 09_train_bi_encoder.py     # Bi-Encoder training
├── 10_build_faiss_index.py    # FAISS index building
├── 11_train_cross_encoder.py  # Cross-Encoder training
└── 12_evaluate_pipeline.py    # Pipeline evaluation
```

### **Data Storage:**
```
data/
├── raw/                    # Original data files
│   ├── legal_corpus.json  # Legal documents corpus
│   └── train.json         # Training data
├── processed/              # Processed data files
│   ├── aid_map.pkl        # Article ID mapping
│   ├── train_triplets_*.jsonl  # Training triplets
│   └── train_pairs_*.jsonl     # Training pairs
└── validation/             # Validation data
```

### **Model Storage:**
```
models/
├── bi-encoder/            # Bi-Encoder model files
│   ├── config.json        # Model configuration
│   ├── pytorch_model.bin  # Model weights
│   └── tokenizer.json     # Tokenizer files
└── cross-encoder-v2/      # Cross-Encoder model files
    ├── config.json        # Model configuration
    ├── pytorch_model.bin  # Model weights
    └── tokenizer.json     # Tokenizer files
```

### **Index Storage:**
```
indexes/
├── legal.faiss            # FAISS index file
└── index_to_aid.json     # Index to Article ID mapping
```

## 🔄 **PIPELINE ORCHESTRATION**

### **Main Pipeline Runner:**
```python
# run_pipeline.py
class LegalQAPipeline:
    def __init__(self):
        self.pipeline_steps = self._define_pipeline_steps()
        self.progress_tracker = ProgressTracker()
    
    def run_pipeline(self, start_step=None):
        # Execute pipeline steps sequentially
        # Handle errors and logging
        # Generate progress reports
```

### **Step Execution:**
```python
def run_step(self, step):
    # 1. Validate step configuration
    # 2. Execute subprocess
    # 3. Monitor output
    # 4. Handle errors
    # 5. Update progress
```

### **Error Handling:**
```python
# Comprehensive error handling
try:
    # Execute step
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise StepExecutionError(result.stderr)
except Exception as e:
    # Log error
    # Update progress
    # Continue or stop based on step importance
```

## 📈 **PERFORMANCE MONITORING**

### **Metrics Tracking:**
```python
# Performance metrics
metrics = {
    'retrieval_time': 50-100ms,
    'reranking_time': 200-500ms,
    'total_response_time': 250-600ms,
    'memory_usage': '2-4GB',
    'gpu_utilization': '60-80%'
}
```

### **Logging System:**
```python
# Unified logging
logger = get_logger(__name__)
logger.info("Processing query: %s", query)
logger.error("Error in retrieval: %s", error)
logger.debug("Retrieval scores: %s", scores)
```

### **Health Checks:**
```python
# System health monitoring
def health_check():
    # Check model availability
    # Check index integrity
    # Check memory usage
    # Check GPU status
    return health_status
```

## 🚀 **DEPLOYMENT ARCHITECTURE**

### **Development Environment:**
```
Local Machine
├── Python Environment
├── GPU Support (optional)
├── Local Data Storage
└── Development Tools
```

### **Production Environment:**
```
Load Balancer
├── Web Server 1
├── Web Server 2
└── Web Server 3
    ├── LawBot Application
    ├── Model Cache
    └── Request Queue
```

### **Docker Deployment:**
```dockerfile
# Multi-stage build
FROM python:3.9-slim as builder
# Install dependencies

FROM python:3.9-slim as runtime
# Copy application
# Set environment variables
# Expose port
```

## 🔒 **SECURITY CONSIDERATIONS**

### **Data Security:**
- ✅ **Encryption**: Sensitive data encryption
- ✅ **Access Control**: Role-based access
- ✅ **Audit Logging**: Comprehensive logging
- ✅ **Data Privacy**: GDPR compliance

### **System Security:**
- ✅ **Input Validation**: Query sanitization
- ✅ **Rate Limiting**: Request throttling
- ✅ **Authentication**: API key management
- ✅ **HTTPS**: Secure communication

## 📊 **SCALABILITY**

### **Horizontal Scaling:**
```
Multiple Instances
├── Shared Model Cache
├── Load Balancer
└── Database Cluster
```

### **Vertical Scaling:**
```
Single Instance
├── More GPU Memory
├── More CPU Cores
└── More RAM
```

### **Caching Strategy:**
```python
# Multi-level caching
cache = {
    'query_cache': LRU(1000),      # Query results
    'model_cache': LRU(10),         # Model instances
    'index_cache': LRU(1),          # FAISS index
}
```

---

**🎯 Kiến trúc này đảm bảo hiệu suất cao, khả năng mở rộng và dễ bảo trì!** 