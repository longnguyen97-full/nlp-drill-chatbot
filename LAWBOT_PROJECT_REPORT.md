# BÁO CÁO ĐỒ ÁN LAWBOOT - HỆ THỐNG AI HỎI ĐÁP PHÁP LUẬT VIỆT NAM
## Phiên bản: v8.3 | Ngày tạo: 2024

---

## MỤC LỤC
1. [Tổng quan và Kiến trúc tổng thể](#1-tổng-quan-và-kiến-trúc-tổng-thể)
2. [Kiến trúc Pipeline 3 tầng](#2-kiến-trúc-pipeline-3-tầng)
3. [Luồng xử lý Training và MLOps](#3-luồng-xử-lý-training-và-mlops)
4. [Luồng xử lý Request User](#4-luồng-xử-lý-request-user)
5. [Giao diện và Báo cáo](#5-giao-diện-và-báo-cáo)
6. [Kỹ thuật Code và Kiến trúc](#6-kỹ-thuật-code-và-kiến-trúc)
7. [Hướng dẫn Vận hành](#7-hướng-dẫn-vận-hành)
8. [Đánh giá và Khuyến nghị](#8-đánh-giá-và-khuyến-nghị)

---

## 1. TỔNG QUAN VÀ KIẾN TRÚC TỔNG THỂ

### 1.1 Giới thiệu dự án

**LawBot** là một hệ thống AI hỏi đáp pháp luật Việt Nam được xây dựng với kiến trúc 3 tầng (3-tier architecture) hiện đại, tích hợp các kỹ thuật machine learning tiên tiến để cung cấp khả năng tìm kiếm và trả lời câu hỏi pháp luật chính xác và hiệu quả.

**Đặc điểm chính:**
- **Kiến trúc 3 tầng** với mỗi tầng được tối ưu hóa cho một nhiệm vụ cụ thể
- **Sử dụng models tiếng Việt chuyên biệt** (PhoBERT, Vietnamese Bi-Encoder)
- **Pipeline xử lý song song** với caching và optimization
- **Hỗ trợ đa ngôn ngữ** (chủ yếu tiếng Việt)
- **Giao diện web** sử dụng Streamlit
- **Real-data only (không synthetic fallback)**: Toàn bộ training sử dụng dữ liệu thật từ `data_processing/run_preparation.py`; nếu thiếu dữ liệu thật, training dừng với lỗi rõ ràng
- **Centralized Paths & Validation**: Tập trung cấu hình đường dẫn và ngưỡng chất lượng tại `config/paths.py` với các hàm `validate_training_data_paths()`, `get_training_data_path()`; `run_workflow.py` tự động xác thực dữ liệu thật trước các stage training và tự động tìm thư mục processed data mới nhất
- **Centralized Model Configuration**: Tập trung cấu hình model tại `config/models.py` với `MODEL_TYPES`, `MODEL_DIRECTORY_MAPPING`, `MODEL_STATUS_KEYS` để đảm bảo tính nhất quán trong naming và mapping
- **Automated Data Freshness Validation**: Workflow tự động kiểm tra tính mới của dữ liệu và re-run `data_preparation` khi cần thiết

### 1.2 Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAWBOOT v8.3                            │
├─────────────────────────────────────────────────────────────────┤
│  🎯 USER INTERFACE LAYER                                      │
│  ├── Streamlit Web App                                        │
│  ├── Search Page (Tìm kiếm & Hỏi đáp)                        │
│  ├── Analysis Page (Phân tích & Báo cáo)                     │
│  └── System Page (Trạng thái & Cấu hình)                     │
├─────────────────────────────────────────────────────────────────┤
│  🚀 PIPELINE LAYER (3-TIER ARCHITECTURE)                      │
│  ├── Tier 1: Bi-Encoder Retrieval                             │
│  ├── Tier 2: Light Reranker                                   │
│  └── Tier 3: Cross-Encoder Ensemble                          │
├─────────────────────────────────────────────────────────────────┤
│  🤖 MODEL LAYER                                               │
│  ├── Vietnamese Bi-Encoder                                    │
│  ├── PhoBERT-based Models                                     │
│  ├── Ensemble Cross-Encoder                                   │
│  └── FAISS Index Engine                                       │
├─────────────────────────────────────────────────────────────────┤
│  📊 DATA & EVALUATION LAYER                                   │
│  ├── Legal Corpus Management                                  │
│  ├── Training Data Processing                                 │
│  ├── Validation Sets                                          │
│  └── Performance Metrics                                      │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ INFRASTRUCTURE LAYER                                      │
│  ├── Configuration Management                                 │
│  ├── Logging & Monitoring                                     │
│  ├── Caching & Optimization                                   │
│  └── Error Handling                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Cấu trúc thư mục dự án

```
LawBot/
├── app/                          # Giao diện người dùng
│   ├── app.py                   # Main application entry point
│   └── pages/                   # Các trang của ứng dụng
│       ├── search.py            # Trang tìm kiếm
│       ├── analysis.py          # Trang phân tích
│       └── system.py            # Trang hệ thống
├── core/                        # Core engine và pipeline
│   ├── pipeline.py              # Main pipeline orchestrator
│   ├── retrieval.py             # Tier 1: Retrieval engine
│   ├── reranking.py             # Tier 2 & 3: Reranking engine
│   ├── datasets/                # Dataset classes
│   ├── transforms/              # Data transformation utilities
│   ├── utils/                   # Utility functions
│   └── progress_tracker.py      # Progress tracking
├── config/                      # Cấu hình hệ thống
│   ├── default.yml              # Cấu hình mặc định
│   ├── loader.py                # Configuration loader
│   ├── paths.py                 # Centralized paths & validation
│   ├── models.py                # Centralized model configuration
│   └── schemas.py               # Pydantic schemas
├── training/                    # Training scripts và engine
│   ├── engine.py                # Training engine
│   ├── base_script.py           # Base training script
│   ├── run_bi_encoder.py        # Bi-encoder training
│   ├── run_light_ranking.py     # Light reranker training
│   ├── run_reranker.py          # Cross-encoder training
│   ├── hpo.py                   # Hyperparameter optimization
│   ├── hard_negative_mining.py  # HNM implementation
│   └── validation_sets.py       # Validation set management
├── data_processing/             # Data preparation
├── evaluation/                  # Evaluation và metrics
├── features/                    # Processed features và models
├── models/                      # Trained models
├── reports/                     # Evaluation reports
├── logs/                        # Log files
└── requirements.txt             # Dependencies
```

### 1.4 Công nghệ sử dụng

**Backend & ML Framework:**
- **Python 3.8+** - Ngôn ngữ chính
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained models
- **SentenceTransformers** - Bi-encoder models
- **FAISS** - Vector similarity search

**Frontend & Web:**
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

**MLOps & Optimization:**
- **Optuna** - Hyperparameter optimization
- **MLflow** - Experiment tracking
- **Contrastive Learning** - Advanced training techniques
- **Hard Negative Mining (HNM)** - Training data enhancement
- **ADAPT** - Domain adaptation techniques
- **Centralized Configuration Management** - Unified model and path configuration

**Data Processing:**
- **Pydantic** - Data validation
- **YAML** - Configuration management
- **JSON/JSONL** - Data formats

### 1.5 Luồng xử lý tổng quan

```
User Query → Web Interface → Pipeline Orchestrator → 3-Tier Processing → Results → UI Display
     ↓              ↓              ↓                    ↓              ↓         ↓
  Input Text   Streamlit App   LegalQAPipeline    Tier1→Tier2→Tier3  Ranked   Interactive
                                                                    Results   Visualization
```

**Các giai đoạn chính:**
1. **Input Processing**: Người dùng nhập câu hỏi qua giao diện web
2. **Pipeline Orchestration**: Hệ thống điều phối xử lý qua 3 tầng
3. **Multi-tier Processing**: Xử lý tuần tự qua retrieval, light reranking, và cross-encoder
4. **Score Aggregation**: Tổng hợp điểm số từ các tầng
5. **Result Ranking**: Sắp xếp kết quả theo điểm số cuối cùng
6. **Output Display**: Hiển thị kết quả với giao diện tương tác

---

## 2. KIẾN TRÚC PIPELINE 3 TẦNG

### 2.1 Tổng quan kiến trúc 3 tầng

Kiến trúc 3 tầng của LawBot được thiết kế để tối ưu hóa hiệu suất và độ chính xác, với mỗi tầng thực hiện một nhiệm vụ cụ thể:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER QUERY                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  🎯 TIER 1: BI-ENCODER RETRIEVAL                              │
│  ├── Model: Vietnamese Bi-Encoder                             │
│  ├── Technique: Contrastive Learning + HNM                    │
│  ├── Purpose: Fast candidate retrieval                        │
│  ├── Performance: ~1000+ docs/second                          │
│  └── Output: Top-K candidates với retrieval scores            │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  ⚡ TIER 2: LIGHT RERANKER                                    │
│  ├── Model: PhoBERT-base-v2 + ADAPT                          │
│  ├── Technique: Independent ADAPT training                    │
│  ├── Purpose: Fast filtering với domain expertise             │
│  ├── Performance: ~500+ docs/second                           │
│  └── Output: Filtered candidates với light reranker scores    │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  🎯 TIER 3: CROSS-ENCODER ENSEMBLE                            │
│  ├── Model: Ensemble (ADAPT-enhanced + Base)                  │
│  ├── Technique: HPO + HNM + Ensemble learning                 │
│  ├── Purpose: Final ranking với high accuracy                 │
│  ├── Performance: ~100+ docs/second                           │
│  └── Output: Final ranked results với cross-encoder scores    │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  🔄 SCORE AGGREGATION & FINAL RANKING                          │
│  ├── Weighted combination của scores từ 3 tầng                │
│  ├── Final score calculation                                  │
│  ├── Result ranking                                           │
│  └── Output formatting                                        │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESULTS                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Chi tiết từng tầng

#### 2.2.1 Tier 1: Bi-Encoder Retrieval

**Mục đích:** Tìm kiếm nhanh các ứng viên ban đầu từ corpus lớn

**Kỹ thuật sử dụng:**
- **Contrastive Learning**: Sử dụng TripletLoss để học biểu diễn
- **Hard Negative Mining (HNM)**: Tự động tìm negative examples khó
- **ADAPT**: Domain adaptation cho pháp luật Việt Nam
- **Vietnamese Bi-Encoder**: Model chuyên biệt cho tiếng Việt

**Luồng xử lý:**
```
Query → Bi-Encoder Encoding → FAISS Search → Candidate Retrieval → Score Assignment
  ↓           ↓                ↓              ↓                   ↓
Input    Embedding Vector   Similarity     Top-K Docs        Retrieval Score
Text     Generation        Search         with AIDs         (0.0 - 1.0)
```

**Code logic (pseudocode):**
```python
class RetrievalEngine:
    def retrieve(self, query, top_k):
        # Encode query thành embedding
        query_embedding = self.bi_encoder.encode(query)
        
        # Tìm kiếm trong FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Lấy thông tin documents
        candidates = []
        for score, idx in zip(scores, indices):
            aid = self.index_to_aid[idx]
            content = self.corpus_content[aid]
            parent_law = self.aid_to_parent_law[aid]
            
            candidates.append({
                'aid': aid,
                'content': content,
                'parent_law_name': parent_law,
                'retrieval_score': score,
                'retrieval_rank': rank
            })
        
        return candidates
```

#### 2.2.2 Tier 2: Light Reranker

**Mục đích:** Lọc nhanh ứng viên từ Tier 1 với domain expertise

**Kỹ thuật sử dụng:**
- **PhoBERT-base-v2**: Model tiếng Việt chuyên biệt
- **Independent ADAPT Training**: Training độc lập với Tier 1
- **HPO**: Hyperparameter optimization
- **HNM**: Hard negative mining

**Luồng xử lý:**
```
Candidates from Tier 1 → Light Reranker → Similarity Scoring → Score Assignment
         ↓                    ↓              ↓                ↓
    Document List      PhoBERT Model    Query-Doc      Light Reranker
    with Content      + ADAPT          Similarity      Score (0.0-1.0)
```

**Code logic (pseudocode):**
```python
class LightReranker:
    def rank_light(self, query, documents):
        # Encode query và documents
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode([doc['content'] for doc in documents])
        
        # Tính cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        # Gán scores
        for doc, sim in zip(documents, similarities):
            doc['light_reranker_score'] = (sim + 1) / 2  # Convert to 0-1
        
        return documents
```

#### 2.2.3 Tier 3: Cross-Encoder Ensemble

**Mục đích:** Xếp hạng chính xác cuối cùng với ensemble learning

**Kỹ thuật sử dụng:**
- **Ensemble Strategy**: Kết hợp ADAPT-enhanced model từ Tier 2 + Base model
- **Weighted Combination**: 70% ADAPT-enhanced + 30% Base model
- **HPO**: Hyperparameter optimization
- **HNM**: Hard negative mining

**Luồng xử lý:**
```
Filtered Candidates → Cross-Encoder Ensemble → Classification Scoring → Final Ranking
         ↓                      ↓                ↓                ↓
    Document List        Ensemble Model    Binary Class      Cross-Encoder
    from Tier 2         (ADAPT + Base)    Prediction        Score (0.0-1.0)
```

**Code logic (pseudocode):**
```python
class CrossEncoderEnsemble:
    def rank_cross(self, query, documents):
        # Tạo sentence pairs
        sentence_pairs = [(query, doc['content']) for doc in documents]
        
        # Dự đoán với ensemble
        scores = []
        for pair in sentence_pairs:
            # ADAPT model prediction
            adapt_score = self.adapt_model.predict(pair)
            
            # Base model prediction  
            base_score = self.base_model.predict(pair)
            
            # Weighted ensemble
            ensemble_score = 0.7 * adapt_score + 0.3 * base_score
            scores.append(ensemble_score)
        
        # Gán scores
        for doc, score in zip(documents, scores):
            doc['cross_encoder_score'] = score
        
        return documents
```

### 2.3 Score Aggregation và Final Ranking

**Mục đích:** Tổng hợp điểm số từ 3 tầng để tạo điểm cuối cùng

**Công thức tính điểm:**
```python
# Khởi tạo với retrieval score
final_score = retrieval_score

# Cộng light reranker score (nếu có)
if use_light_ranking:
    final_score = light_weight * light_score + (1 - light_weight) * final_score

# Cộng cross encoder score (nếu có)  
if use_cross_encoder:
    final_score = cross_weight * cross_score + (1 - cross_weight) * final_score

# Weights mặc định:
# light_weight = 0.7 (70%)
# cross_weight = 0.3 (30%)
```

**Luồng xử lý:**
```
Scores from 3 Tiers → Weighted Combination → Score Normalization → Final Ranking
         ↓                    ↓                    ↓                ↓
Retrieval + Light +    Weighted Average    Score Range      Sort by Final
Cross Encoder         (Configurable)       0.0 - 1.0        Score Desc
```

**Code logic (pseudocode):**
```python
def combine_scores(documents, use_light=True, use_cross=True):
    for doc in documents:
        retrieval_score = doc.get('retrieval_score', 0.0)
        light_score = doc.get('light_reranker_score', 0.0)
        cross_score = doc.get('cross_encoder_score', 0.0)
        
        # Khởi tạo với retrieval score
        final_score = retrieval_score
        
        # Cộng light reranker
        if use_light:
            final_score = 0.7 * light_score + 0.3 * final_score
        
        # Cộng cross encoder
        if use_cross:
            final_score = 0.3 * cross_score + 0.7 * final_score
        
        doc['final_score'] = final_score
        doc['score_breakdown'] = {
            'retrieval': retrieval_score,
            'light_reranker': light_score,
            'cross_encoder': cross_score
        }
    
    # Sắp xếp theo điểm cuối cùng
    return sorted(documents, key=lambda x: x['final_score'], reverse=True)
```

### 2.4 Pipeline Orchestration

**Mục đích:** Điều phối luồng xử lý qua 3 tầng một cách hiệu quả

**Luồng điều phối:**
```
Query Input → Pipeline Orchestrator → Tier 1 → Tier 2 → Tier 3 → Score Aggregation → Results
     ↓              ↓                ↓        ↓        ↓        ↓                ↓
User Question   LegalQAPipeline   Retrieval  Light    Cross    Combine          Final
                .predict()        Engine     Rerank   Encode   Scores           Output
```

**Code logic (pseudocode):**
```python
class LegalQAPipeline:
    def predict(self, query, top_k_retrieval=100, top_k_final=10):
        # Tier 1: Bi-Encoder Retrieval
        documents = self.retriever.retrieve(query, top_k=top_k_retrieval)
        
        # Khởi tạo scores
        for doc in documents:
            doc['light_reranker_score'] = 0.0
            doc['cross_encoder_score'] = 0.0
        
        # Tier 2: Light Reranking (nếu enabled)
        if self.use_light_ranking and self.reranker.is_ready:
            documents = self.reranker.rank_light(query, documents[:top_k_light])
        
        # Tier 3: Cross-Encoder Reranking (nếu enabled)
        if self.use_cross_encoder and self.reranker.is_ready:
            documents = self.reranker.rank_cross(query, documents)
        
        # Tổng hợp scores và sắp xếp
        documents = self._combine_scores(documents)
        final_results = sorted(documents, key=lambda x: x['final_score'], reverse=True)[:top_k_final]
        
        return final_results
```

---

## 3. LUỒNG XỬ LÝ TRAINING VÀ MLOPS

### 3.1 Tổng quan luồng training

Luồng training của LawBot được thiết kế theo kiến trúc 3 tầng, với mỗi tầng có quy trình training riêng biệt và tối ưu hóa:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│  📊 DATA PREPARATION STAGE                                    │
│  ├── Raw Data Loading                                         │
│  ├── Legal Corpus Processing                                  │
│  ├── Training Data Generation                                 │
│  ├── Validation Set Creation                                  │
│  └── Feature Engineering                                      │
├─────────────────────────────────────────────────────────────────┤
│  🎯 TIER 1 TRAINING: BI-ENCODER                              │
│  ├── Contrastive Learning                                     │
│  ├── Hard Negative Mining (HNM)                               │
│  ├── ADAPT Domain Adaptation                                  │
│  ├── Hyperparameter Optimization (HPO)                        │
│  └── Model Export & FAISS Index Creation                      │
├─────────────────────────────────────────────────────────────────┤
│  ⚡ TIER 2 TRAINING: LIGHT RERANKER                           │
│  ├── PhoBERT-base-v2 Fine-tuning                              │
│  ├── Independent ADAPT Training                                │
│  ├── HNM Implementation                                       │
│  ├── HPO Optimization                                         │
│  └── Model Export                                             │
├─────────────────────────────────────────────────────────────────┤
│  🎯 TIER 3 TRAINING: CROSS-ENCODER ENSEMBLE                   │
│  ├── Ensemble Model Creation                                  │
│  ├── ADAPT-enhanced Model Integration                         │
│  ├── HPO + HNM Optimization                                   │
│  ├── Ensemble Training                                        │
│  └── Model Export                                             │
├─────────────────────────────────────────────────────────────────┤
│  🔄 PIPELINE INTEGRATION & VALIDATION                         │
│  ├── Pipeline Assembly                                        │
│  ├── End-to-end Testing                                       │
│  ├── Performance Evaluation                                   │
│  ├── Comprehensive Metrics                                    │
│  └── Deployment Preparation                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Preparation Stage

#### 3.2.1 Raw Data Loading và Processing

**Mục đích:** Chuẩn bị dữ liệu thô cho training pipeline

**Luồng xử lý:**
```
Raw Legal Documents → Text Extraction → Cleaning → Normalization → Structured Data
         ↓                ↓              ↓          ↓              ↓
    PDF/DOC Files    Content Parse   Legal Text   Unicode      Training
    Legal Corpus     Article Split   Cleaning     Normalize    Examples
```

**Code logic (pseudocode):**
```python
def process_legal_corpus(raw_corpus):
    processed_corpus = {}
    aid_map = {}
    
    for document in raw_corpus:
        # Extract content và metadata
        content = document.get('content', '')
        article_id = document.get('article_id')
        law_id = document.get('law_id')
        
        # Clean và normalize text
        cleaned_content = legal_text_cleaner(content)
        
        # Generate canonical AID
        canonical_aid = canonicalize_aid(law_id, article_id)
        
        # Store processed data
        processed_corpus[canonical_aid] = cleaned_content
        aid_map[canonical_aid] = {
            'law_id': law_id,
            'article_id': article_id,
            'content': cleaned_content
        }
    
    return processed_corpus, aid_map
```

#### 3.2.2 Training Data Generation

**Mục đích:** Tạo training examples cho từng tier với kỹ thuật khác nhau

**Luồng xử lý:**
```
Legal Corpus → Query Generation → Positive/Negative Mining → Training Examples
      ↓              ↓                    ↓                    ↓
Article Content   Question Types    Similarity-based     Structured Data
+ Metadata       + Templates       + Hard Negatives     for Training
```

**Code logic (pseudocode):**
```python
def generate_training_examples(train_data, processed_corpus, aid_map):
    # Tier 1: Bi-encoder training data
    bi_encoder_data = []
    for item in train_data:
        query = item['question']
        positive_aid = item['positive_aid']
        negative_aids = item['negative_aids']
        
        bi_encoder_data.append({
            'query': query,
            'positive': processed_corpus[positive_aid],
            'negative': processed_corpus[negative_aids[0]]  # Use first negative
            'positive_aid': positive_aid,
            'negative_aid': negative_aids[0]
        })
    
    # Tier 2 & 3: Reranker training data
    reranker_data = []
    for item in train_data:
        query = item['question']
        positive_aid = item['positive_aid']
        negative_aids = item['negative_aids']
        
        # Positive examples
        reranker_data.append({
            'query': query,
            'passage': processed_corpus[positive_aid],
            'label': 1.0,
            'aid': positive_aid
        })
        
        # Negative examples
        for neg_aid in negative_aids[:3]:  # Top 3 negatives
            reranker_data.append({
                'query': query,
                'passage': processed_corpus[neg_aid],
                'label': 0.0,
                'aid': neg_aid
            })
    
    return bi_encoder_data, reranker_data
```

#### 3.2.3 Validation Set Creation

**Mục đích:** Tạo validation sets riêng biệt cho từng tier

**Luồng xử lý:**
```
Training Data → Stratified Split → Tier-specific Validation → Validation Files
      ↓              ↓                    ↓                    ↓
Full Dataset    Train/Val Split    Tier 1/2/3 Sets      JSONL Files
                (80/20)           with Metrics          for Evaluation
```

**Code logic (pseudocode):**
```python
class ValidationSetManager:
    def create_tier_validation_sets(self, training_data, validation_split=0.2):
        # Split data
        train_data, val_data = train_test_split(
            training_data, test_size=validation_split, random_state=42
        )
        
        # Tier 1: Retrieval validation (query-document pairs)
        tier1_val = self._create_retrieval_validation(val_data, min_samples=100)
        
        # Tier 2: Light reranker validation (classification format)
        tier2_val = self._create_light_reranker_validation(val_data, min_samples=100)
        
        # Tier 3: Cross-encoder validation (reranking format)
        tier3_val = self._create_cross_encoder_validation(val_data, min_samples=100)
        
        return tier1_val, tier2_val, tier3_val
```

### 3.3 Tier 1 Training: Bi-Encoder

#### 3.3.1 Contrastive Learning Implementation

**Mục đích:** Huấn luyện bi-encoder sử dụng contrastive learning với TripletLoss

**Kỹ thuật sử dụng:**
- **TripletLoss**: (query, positive, negative) training
- **Hard Negative Mining**: Tự động tìm negative examples khó
- **ADAPT**: Domain adaptation cho pháp luật Việt Nam
- **HPO**: Hyperparameter optimization

**Luồng training:**
```
Training Data → Triplet Creation → Model Forward → Loss Calculation → Backpropagation
      ↓              ↓              ↓              ↓                ↓
Query+Pos+Neg   Triplet Batching   Embedding     TripletLoss      Gradient
Examples        with HNM          Generation     Computation      Update
```

**Code logic (pseudocode):**
```python
class BiEncoderTrainer:
    def train_contrastive_learning(self, model_name):
        # Load pre-trained Vietnamese bi-encoder
        model = SentenceTransformer(model_name)
        
        # Prepare contrastive data
        triplets = self._prepare_contrastive_data()
        
        # Custom training loop
        for epoch in range(self.epochs):
            for batch in self.triplet_dataloader:
                # Forward pass
                query_emb = model.encode(batch['queries'])
                pos_emb = model.encode(batch['positives'])
                neg_emb = model.encode(batch['negatives'])
                
                # Calculate triplet loss
                loss = self.triplet_loss(query_emb, pos_emb, neg_emb)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return model
```

#### 3.3.2 Hard Negative Mining (HNM)

**Mục đích:** Tự động tìm negative examples khó để cải thiện training

**Luồng xử lý:**
```
Candidate Pool → Similarity Calculation → Hard Negative Selection → Training Update
      ↓                ↓                      ↓                    ↓
All Documents    Query-Doc Similarity    Top-K Difficult     Enhanced
                 with Current Model      Negatives          Training Data
```

**Code logic (pseudocode):**
```python
class HardNegativeMiner:
    def mine_hard_negatives(self, queries, positive_candidates, negative_candidates):
        hard_negatives = []
        
        for query in queries:
            # Encode query và candidates
            query_emb = self.model.encode(query)
            pos_embs = self.model.encode(positive_candidates)
            neg_embs = self.model.encode(negative_candidates)
            
            # Calculate similarities
            pos_sims = cosine_similarity(query_emb, pos_embs)
            neg_sims = cosine_similarity(query_emb, neg_embs)
            
            # Find hardest negatives (highest similarity to query)
            hardest_neg_indices = np.argsort(neg_sims)[-self.top_k:]
            
            # Select hardest negatives
            for idx in hardest_neg_indices:
                if neg_sims[idx] > self.similarity_threshold:
                    hard_negatives.append(negative_candidates[idx])
        
        return hard_negatives
```

#### 3.3.3 ADAPT Domain Adaptation

**Mục đích:** Thích ứng model với domain pháp luật Việt Nam

**Luồng xử lý:**
```
Base Model → Domain Data Loading → ADAPT Training → Domain-adapted Model
     ↓              ↓                ↓                ↓
Pre-trained    Legal Corpus     Fine-tuning      Optimized for
Vietnamese     + Queries        with ADAPT       Legal Domain
Bi-encoder
```

**Code logic (pseudocode):**
```python
def apply_adapt_technique(self, model, domain_data):
    # Prepare domain-specific dataset
    domain_dataset = LegalDomainDataset(domain_data)
    domain_dataloader = DataLoader(domain_dataset, batch_size=16)
    
    # ADAPT training loop
    for epoch in range(self.adaptation_steps):
        for batch in domain_dataloader:
            # Forward pass với domain data
            outputs = model(batch['input_ids'], batch['attention_mask'])
            
            # Calculate domain adaptation loss
            loss = self.domain_loss(outputs, batch['labels'])
            
            # Backward pass
            loss.backward()
            self.adapt_optimizer.step()
            self.adapt_optimizer.zero_grad()
    
    return model
```

### 3.4 Tier 2 Training: Light Reranker

#### 3.4.1 PhoBERT Fine-tuning

**Mục đích:** Huấn luyện PhoBERT-base-v2 cho light reranking

**Kỹ thuật sử dụng:**
- **PhoBERT-base-v2**: Base model tiếng Việt
- **Independent ADAPT Training**: Training độc lập với Tier 1
- **Classification Head**: Binary classification cho relevance
- **HPO**: Hyperparameter optimization

**Luồng training:**
```
PhoBERT Model → Data Loading → Fine-tuning → HPO Optimization → Model Export
      ↓              ↓              ↓              ↓                ↓
Pre-trained    Training Data   Classification   Optuna Trials    Saved Model
PhoBERT        + Labels        Training         + Best Params     + Config
```

**Code logic (pseudocode):**
```python
class LightRankingTrainer:
    def train_model(self, data_path, hpo_params=None):
        # Load PhoBERT model và tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base-v2", num_labels=1
        )
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        
        # Prepare training data
        train_dataset, val_dataset = self._load_training_data(data_path)
        
        # Training loop
        for epoch in range(self.epochs):
            for batch in self.train_dataloader:
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate loss
                loss = self.criterion(outputs.logits, batch['labels'])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return model, tokenizer
```

#### 3.4.2 Hyperparameter Optimization (HPO)

**Mục đích:** Tối ưu hóa hyperparameters sử dụng Optuna

**Luồng optimization:**
```
HPO Configuration → Trial Generation → Model Training → Evaluation → Best Params
        ↓                ↓                ↓              ↓            ↓
Parameter Space    Optuna Trials    Train Model    Metrics      Optimal
Definition        + Sampling       with Params    Calculation   Parameters
```

**Code logic (pseudocode):**
```python
class HyperparameterOptimizer:
    def objective(self, trial):
        # Define hyperparameter space
        params = {
            'learning_rate': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'epochs': trial.suggest_int('epochs', 2, 5),
            'warmup_steps': trial.suggest_int('warmup_steps', 50, 200),
            'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.1)
        }
        
        # Train model với params
        model = self._train_with_params(params)
        
        # Evaluate model
        val_score = self._evaluate_model(model)
        
        return val_score
    
    def optimize(self, n_trials=15):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return study.best_params
```

### 3.5 Tier 3 Training: Cross-Encoder Ensemble

#### 3.5.1 Ensemble Model Creation

**Mục đích:** Tạo ensemble model kết hợp ADAPT-enhanced và base model

**Kỹ thuật sử dụng:**
- **Ensemble Strategy**: Weighted combination (70% ADAPT + 30% Base)
- **Model Integration**: PhoBERT-base-v2 (ADAPT) + PhoBERT-large (Base)
- **HPO**: Hyperparameter optimization cho ensemble weights
- **HNM**: Hard negative mining

**Luồng training:**
```
ADAPT Model → Base Model → Ensemble Creation → Joint Training → Model Export
     ↓            ↓              ↓                ↓            ↓
Tier 2 Output   PhoBERT-large   Weighted        Fine-tuning   Ensemble
PhoBERT-base    Pre-trained     Combination     + HPO          Model
```

**Code logic (pseudocode):**
```python
class CrossEncoderTrainer:
    def create_ensemble_strategy(self, adapt_model, base_model):
        # Create ensemble model
        ensemble = EnsembleReranker(
            adapt_model=adapt_model,
            base_model=base_model,
            adapt_weight=0.7,
            base_weight=0.3
        )
        
        # Prepare training data với HNM
        training_data = self._create_reranking_training_data()
        enriched_data = self._enrich_with_hard_negatives(training_data)
        
        # Training loop
        for epoch in range(self.epochs):
            for batch in self.train_dataloader:
                # Forward pass với ensemble
                outputs = ensemble(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate loss
                loss = self.criterion(outputs.logits, batch['labels'])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return ensemble
```

#### 3.5.2 Ensemble Model Architecture

**Mục đích:** Thiết kế kiến trúc ensemble cho cross-encoder

**Kiến trúc:**
```
Input (Query + Document) → Tokenization → Ensemble Processing → Final Output
         ↓                      ↓                ↓                ↓
Text Pair              Token IDs        Parallel Models    Weighted Score
                       + Attention      (ADAPT + Base)    Combination
```

**Code logic (pseudocode):**
```python
class EnsembleReranker(torch.nn.Module):
    def __init__(self, adapt_model, base_model, adapt_weight=0.7, base_weight=0.3):
        super().__init__()
        self.adapt_model = adapt_model
        self.base_model = base_model
        self.adapt_weight = adapt_weight
        self.base_weight = base_weight
    
    def forward(self, input_ids, attention_mask):
        # Get predictions from both models
        with torch.no_grad():
            adapt_output = self.adapt_model(input_ids, attention_mask)
            base_output = self.base_model(input_ids, attention_mask)
        
        # Weighted combination
        ensemble_logits = (
            self.adapt_weight * adapt_output.logits +
            self.base_weight * base_output.logits
        )
        
        return SequenceClassifierOutput(logits=ensemble_logits)
```

### 3.6 MLOps và Pipeline Management

#### 3.6.1 Workflow Orchestration với Centralized Paths

**Mục đích:** Quản lý và điều phối toàn bộ training workflow với centralized path management

**Luồng workflow:**
```
Project Setup → Data Preparation → Model Training → Evaluation → Deployment
      ↓              ↓                ↓              ↓            ↓
Environment     Legal Corpus     Tier 1/2/3      Metrics      Model
Configuration   Processing       Training        Calculation   Serving
```

**Centralized Path Management:**
```python
# config/paths.py - Centralized configuration
TRAINING_DATA_PATHS = {
    "primary": {
        "bi_encoder": Path("features/processed_data/bi_encoder_train.jsonl"),
        "cross_encoder": Path("features/processed_data/cross_encoder_train.jsonl"),
        "processed_corpus": Path("features/processed_data/processed_corpus.json"),
        "training_data": Path("features/processed_data/training_data.jsonl"),
        "negative_pool": Path("features/processed_data/negative_pool.jsonl"),
    },
    "validation": {
        "tier_1": Path("features/validation_sets/tier_1_validation.jsonl"),
        "tier_2": Path("features/validation_sets/tier_2_validation.jsonl"),
        "tier_3": Path("features/validation_sets/tier_3_validation.jsonl"),
    }
}

# Automated data freshness validation
def validate_training_data_paths() -> dict:
    """Automatically find latest processed data directory"""
    features_dir = Path("features")
    processed_dirs = [d for d in features_dir.iterdir() 
                     if d.is_dir() and d.name.startswith("processed_data_")]
    
    if processed_dirs:
        latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
        return {"latest_data_dir": str(latest_dir), "status": "ready"}
    return {"status": "missing_data"}
```

**Code logic (pseudocode):**
```python
class WorkflowRunner:
    def run_workflow(self):
        stages = [
            'data_preparation',
            'bi_encoder_training',
            'faiss_index_creation',
            'light_reranker_training',
            'cross_encoder_training',
            'pipeline_integration',
            'evaluation'
        ]
        
        for stage in stages:
            # Special handling for data_preparation with freshness validation
            if stage == "data_preparation":
                success = self._run_data_preparation_with_validation()
            else:
                success = self.run_stage(stage)
            
            if not success:
                self._handle_failure(stage)
                break
        
        return success
    
    def _run_data_preparation_with_validation(self):
        """Run data preparation with automatic freshness validation"""
        try:
            from config.paths import validate_training_data_paths
            validation_results = validate_training_data_paths()
            
            if validation_results["overall"]["real_data_available"]:
                self.logger.info("✅ Real training data available, skipping data_preparation")
                return True
            else:
                self.logger.info("⚠️ Data stale/missing, running data_preparation...")
                return self.run_stage("data_preparation")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Data validation failed: {e}, running data_preparation...")
            return self.run_stage("data_preparation")
```

#### 3.6.2 Model Versioning và Management

**Mục đích:** Quản lý phiên bản models và tracking experiments

**Luồng versioning:**
```
Training Completion → Model Export → Metadata Generation → Version Tracking → Registry
         ↓                ↓                ↓                ↓            ↓
Trained Model      Save to Disk      Model Info        Timestamp     Model
                   + Config          + Metrics         + Hash        Registry
```

**Code logic (pseudocode):**
```python
def save_model(self, engine, output_dir):
    # Generate versioned path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = output_dir / f"model_{timestamp}"
    
    # Save model
    engine.save_pretrained(versioned_path)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': self.model_type,
        'training_params': self.training_params,
        'performance_metrics': self.evaluation_results,
        'git_commit': self._get_git_commit()
    }
    
    with open(versioned_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return versioned_path
```

#### 3.6.3 Performance Monitoring và Evaluation

**Mục đích:** Theo dõi hiệu suất training và đánh giá models

**Luồng monitoring:**
```
Training Progress → Metrics Collection → Performance Analysis → Report Generation
        ↓                ↓                    ↓                ↓
Real-time Logs    Loss/Accuracy      Tier Comparison     Evaluation
+ Checkpoints     + Validation       + Improvement       Reports
```

**Code logic (pseudocode):**
```python
class ProgressTracker:
    def start_step(self, step_name, step_info=None):
        self.current_step += 1
        step_start_time = time.time()
        
        # Log step start
        self.logger.info(f"[{self.current_step}/{self.total_steps}] Starting: {step_name}")
        
        # Store step data
        step_data = {
            'name': step_name,
            'info': step_info,
            'start_time': step_start_time,
            'status': 'running'
        }
        self.steps_info.append(step_data)
        
        return step_start_time
    
    def end_step(self, step_start_time, success=True):
        end_time = time.time()
        duration = end_time - step_start_time
        
        # Update step data
        if self.steps_info:
            step_data = self.steps_info[-1]
            step_data.update({
                'end_time': end_time,
                'duration': duration,
                'success': success,
                'status': 'completed' if success else 'failed'
            })
        
        # Log completion
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Step {step_data['name']} {status} in {duration:.2f}s")
```

### 3.7 Training Configuration và Optimization

#### 3.7.1 Configuration Management

**Mục đích:** Quản lý cấu hình training cho từng tier

**Cấu trúc config:**
```yaml
# Training configuration cho từng tier
bi_encoder:
  model_name: "bkai-foundation-models/vietnamese-bi-encoder"
  enhancement_techniques:
    contrastive_learning:
      enabled: true
      loss_type: "TripletLoss"
      max_length: 256
      batch_size: 16
      epochs: 5
    adapt:
      enabled: true
      domain_data_ratio: 0.8
      adaptation_steps: 1000
      learning_rate: 2e-5

light_reranker:
  model_name: "vinai/phobert-base-v2"
  max_length: 256
  batch_size: 32
  epochs: 3
  learning_rate: 2e-5

cross_encoder:
  ensemble_method: "weighted_average"
  models:
    adapt_enhanced: 0.7
    base_model: 0.3
```

#### 3.7.2 Training Optimization Techniques

**Mục đích:** Tối ưu hóa quá trình training cho hiệu suất cao

**Kỹ thuật optimization:**
- **Gradient Accumulation**: Xử lý batch lớn với memory hạn chế
- **Mixed Precision Training**: Sử dụng FP16 để tăng tốc
- **Learning Rate Scheduling**: Warmup và decay strategies
- **Early Stopping**: Tránh overfitting
- **Model Checkpointing**: Lưu trữ intermediate models

**Code logic (pseudocode):**
```python
class TrainingEngine:
    def setup_optimizer_and_scheduler(self, num_training_steps):
        # Optimizer với weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler với warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, epoch):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass với gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
```

---

## 4. LUỒNG XỬ LÝ REQUEST USER

### 4.1 Tổng quan luồng xử lý request

Luồng xử lý request của LawBot được thiết kế để xử lý câu hỏi pháp luật từ người dùng một cách hiệu quả và chính xác:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER REQUEST FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│  🎯 USER INPUT STAGE                                          │
│  ├── Query Input (Text/Question)                              │
│  ├── Search Configuration                                     │
│  ├── Parameter Selection                                      │
│  └── Request Validation                                       │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ REQUEST PROCESSING STAGE                                  │
│  ├── Query Preprocessing                                      │
│  ├── Parameter Calculation                                    │
│  ├── Pipeline Initialization                                  │
│  └── Cache Management                                         │
├─────────────────────────────────────────────────────────────────┤
│  🚀 PIPELINE EXECUTION STAGE                                  │
│  ├── Tier 1: Bi-Encoder Retrieval                             │
│  ├── Tier 2: Light Reranking                                  │
│  ├── Tier 3: Cross-Encoder Reranking                          │
│  └── Score Aggregation                                        │
├─────────────────────────────────────────────────────────────────┤
│  📊 RESULT PROCESSING STAGE                                   │
│  ├── Result Ranking                                           │
│  ├── Score Normalization                                      │
│  ├── Metadata Enrichment                                      │
│  └── Response Formatting                                       │
├─────────────────────────────────────────────────────────────────┤
│  🎨 USER OUTPUT STAGE                                         │
│  ├── Interactive Visualization                                 │
│  ├── Score Comparison Charts                                  │
│  ├── Detailed Result Display                                  │
│  └── Performance Metrics                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 User Input Stage

#### 4.2.1 Query Input và Validation

**Mục đích:** Xử lý và validate input từ người dùng

**Luồng xử lý:**
```
User Input → Text Validation → Query Preprocessing → Search Configuration → Request Ready
     ↓              ↓                ↓                    ↓                ↓
Question Text   Input Check      Text Cleaning      Parameter Set    Validated
+ Parameters   + Sanitization   + Normalization    + Options        Request
```

**Code logic (pseudocode):**
```python
def process_user_query(query_input, search_config):
    # Validate input
    if not query_input or len(query_input.strip()) < 3:
        raise ValueError("Query must be at least 3 characters long")
    
    # Clean và normalize text
    cleaned_query = legal_text_cleaner(query_input.strip())
    
    # Extract search parameters
    final_results_count = search_config.get('final_results_count', 3)
    search_aggressiveness = search_config.get('search_aggressiveness', 'balanced')
    force_cpu = search_config.get('force_cpu', False)
    
    # Validate parameters
    if final_results_count < 1 or final_results_count > 50:
        raise ValueError("Final results count must be between 1 and 50")
    
    return {
        'query': cleaned_query,
        'final_results_count': final_results_count,
        'search_aggressiveness': search_aggressiveness,
        'force_cpu': force_cpu
    }
```

#### 4.2.2 Search Configuration Management

**Mục đích:** Quản lý cấu hình tìm kiếm và tối ưu hóa parameters

**Các loại cấu hình:**
- **Conservative**: Ít kết quả, chính xác cao
- **Balanced**: Cân bằng giữa số lượng và chất lượng
- **Aggressive**: Nhiều kết quả, đa dạng

**Code logic (pseudocode):**
```python
def calculate_optimal_parameters(final_results_count, search_aggressiveness):
    # Multiplier dựa trên mức độ tìm kiếm
    search_multipliers = {
        'conservative': {'retrieval': 2, 'light': 1.5},
        'balanced': {'retrieval': 3, 'light': 2},
        'aggressive': {'retrieval': 5, 'light': 3}
    }
    
    mult = search_multipliers.get(search_aggressiveness, search_multipliers['balanced'])
    
    # Tier 1: Retrieval - lấy nhiều candidates cho Tier 2
    top_k_retrieval = max(
        config.app.top_k_retrieval,
        final_results_count * mult['retrieval']
    )
    
    # Tier 2: Light Reranking - đủ cho Tier 3 xử lý
    top_k_light_reranking = max(20, final_results_count * mult['light'])
    
    return {
        'top_k_retrieval': top_k_retrieval,
        'top_k_light_reranking': top_k_light_reranking,
        'top_k_final': final_results_count
    }
```

### 4.3 Request Processing Stage

#### 4.3.1 Query Preprocessing

**Mục đích:** Chuẩn bị query cho pipeline processing

**Luồng xử lý:**
```
Raw Query → Text Cleaning → Legal Text Processing → Query Enhancement → Pipeline Ready
     ↓            ↓                ↓                    ↓                ↓
User Input   Remove Noise     Legal-specific      Context Info     Processed
Text         + Normalize      Cleaning            + Metadata       Query
```

**Code logic (pseudocode):**
```python
class LegalTextCleaner(TextProcessor):
    def __call__(self, text):
        # Base processing
        text = super().__call__(text)
        
        # Legal-specific cleaning
        # Remove article/clause numbers
        text = self.article_pattern.sub(r"\1", text)
        
        # Replace common abbreviations
        for abbr, full_text in self.abbreviation_map.items():
            text = re.sub(rf"\b{abbr}\b", full_text, text)
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

def preprocess_query(query):
    # Initialize legal text cleaner
    cleaner = LegalTextCleaner()
    
    # Clean query
    cleaned_query = cleaner(query)
    
    # Add context information
    query_context = {
        'original_query': query,
        'cleaned_query': cleaned_query,
        'query_length': len(cleaned_query),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    return cleaned_query, query_context
```

#### 4.3.2 Pipeline Initialization và Cache Management

**Mục đích:** Khởi tạo pipeline và quản lý cache để tối ưu hiệu suất

**Luồng xử lý:**
```
Request → Cache Check → Pipeline Load → Model Validation → Ready for Processing
    ↓          ↓            ↓              ↓                ↓
User Query   Cache Hit?   Load Models   Validate Status   Pipeline Ready
             (Yes/No)     + Config      + Health Check     for Execution
```

**Code logic (pseudocode):**
```python
@st.cache_resource(ttl=config.app.cache_pipeline_ttl_seconds)
def load_pipeline(force_cpu=False):
    try:
        # Set CUDA environment variables
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Force CPU nếu cần
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Memory cleanup
        gc.collect()
        
        # Load pipeline
        start_time = time.time()
        pipeline = LegalQAPipeline()
        load_time = time.time() - start_time
        
        # Validate pipeline
        if not pipeline.is_ready:
            raise RuntimeError("Pipeline initialization failed")
        
        # Validate parent law mapping
        ensure_parent_law_mapping()
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Pipeline loading failed: {e}")
        raise

def get_pipeline_lazy():
    """Lazy load pipeline chỉ khi cần thiết"""
    try:
        # Validate parent law mapping
        if ensure_parent_law_mapping():
            logger.info("✅ Parent law mapping validated")
        
        pipeline = LegalQAPipeline()
        return pipeline
        
    except Exception as e:
        logger.warning(f"⚠️ Pipeline not available: {e}")
        return None
```

### 4.4 Pipeline Execution Stage

#### 4.4.1 Tier 1: Bi-Encoder Retrieval

**Mục đích:** Tìm kiếm nhanh candidates ban đầu từ corpus

**Luồng xử lý:**
```
Query → Bi-Encoder → Embedding → FAISS Search → Candidate Retrieval → Score Assignment
  ↓         ↓           ↓           ↓              ↓                ↓
Input   Model      Vector      Similarity      Top-K Docs      Retrieval
Text    Encoding   Generation  Search         with AIDs       Score
```

**Code logic (pseudocode):**
```python
def execute_tier1_retrieval(pipeline, query, top_k_retrieval):
    logger.info("🎯 Tier 1: Bi-Encoder Retrieval")
    
    try:
        # Get documents từ retriever
        documents = pipeline.retriever.retrieve(query, top_k=top_k_retrieval)
        
        if not documents:
            logger.warning("No documents retrieved from Bi-Encoder")
            return []
        
        # Initialize scores cho các tiers tiếp theo
        for doc in documents:
            doc['light_reranker_score'] = 0.0
            doc['cross_encoder_score'] = 0.0
        
        logger.info(f"✅ Tier 1 completed: {len(documents)} candidates retrieved")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Tier 1 failed: {e}")
        return []
```

#### 4.4.2 Tier 2: Light Reranking

**Mục đích:** Lọc nhanh candidates với domain expertise

**Luồng xử lý:**
```
Candidates → Light Reranker → Similarity Scoring → Score Assignment → Filtered Results
     ↓            ↓                ↓                ↓                ↓
Document List   PhoBERT Model   Query-Doc      Light Reranker   Enhanced
from Tier 1    + ADAPT         Similarity     Score (0.0-1.0)  Candidates
```

**Code logic (pseudocode):**
```python
def execute_tier2_light_reranking(pipeline, query, documents, top_k_light):
    logger.info("⚡ Tier 2: Light Reranking")
    
    try:
        if not pipeline.reranker or not pipeline.reranker.is_ready:
            logger.warning("⚠️ Light reranking skipped: reranker not ready")
            return documents
        
        # Light reranking với top-k candidates
        documents = pipeline.reranker.rank_light(query, documents[:top_k_light])
        
        # Debug: Check scores
        for doc in documents[:3]:
            logger.info(f"🔍 After light reranking: light_score={doc.get('light_reranker_score', 'N/A')}")
        
        logger.info(f"✅ Tier 2 completed: {len(documents)} candidates reranked")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Tier 2 failed: {e}")
        return documents
```

#### 4.4.3 Tier 3: Cross-Encoder Reranking

**Mục đích:** Xếp hạng chính xác cuối cùng với ensemble model

**Luồng xử lý:**
```
Filtered Candidates → Cross-Encoder → Classification → Score Assignment → Final Ranking
         ↓                ↓              ↓              ↓                ↓
Document List      Ensemble Model   Binary Class    Cross-Encoder    Ranked
from Tier 2       (ADAPT + Base)   Prediction      Score (0.0-1.0)  Results
```

**Code logic (pseudocode):**
```python
def execute_tier3_cross_encoder(pipeline, query, documents):
    logger.info("🎯 Tier 3: Cross-Encoder Reranking")
    
    try:
        if not pipeline.reranker or not pipeline.reranker.is_ready:
            logger.warning("⚠️ Cross-encoder reranking skipped: reranker not ready")
            return documents
        
        # Cross-encoder reranking
        documents = pipeline.reranker.rank_cross(query, documents)
        
        # Debug: Check scores
        for doc in documents[:3]:
            logger.info(f"🔍 After cross-encoder: cross_score={doc.get('cross_encoder_score', 'N/A')}")
        
        logger.info(f"✅ Tier 3 completed: {len(documents)} candidates reranked")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Tier 3 failed: {e}")
        return documents
```

#### 4.4.4 Score Aggregation và Final Ranking

**Mục đích:** Tổng hợp scores từ 3 tiers và sắp xếp kết quả cuối cùng

**Luồng xử lý:**
```
Scores from 3 Tiers → Weighted Combination → Score Normalization → Final Ranking
         ↓                    ↓                    ↓                ↓
Retrieval + Light +    Weighted Average    Score Range      Sort by Final
Cross Encoder         (Configurable)       0.0 - 1.0        Score Desc
```

**Code logic (pseudocode):**
```python
def aggregate_scores_and_rank(documents, use_light=True, use_cross=True):
    logger.info("🔄 Combining scores from all tiers...")
    
    try:
        for doc in documents:
            retrieval_score = doc.get('retrieval_score', 0.0)
            light_score = doc.get('light_reranker_score', 0.0)
            cross_score = doc.get('cross_encoder_score', 0.0)
            
            # Flexible score combination
            final_score = retrieval_score
            
            # Add light reranker score
            if use_light:
                final_score = 0.7 * light_score + 0.3 * final_score
            
            # Add cross encoder score
            if use_cross:
                final_score = 0.3 * cross_score + 0.7 * final_score
            
            # Store final score và breakdown
            doc['final_score'] = final_score
            doc['score_breakdown'] = {
                'retrieval': retrieval_score,
                'light_reranker': light_score,
                'cross_encoder': cross_score
            }
        
        # Sort by final score
        final_results = sorted(documents, key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"✅ Score aggregation completed: {len(final_results)} results ranked")
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Score aggregation failed: {e}")
        return documents
```

### 4.5 Result Processing Stage

#### 4.5.1 Result Ranking và Normalization

**Mục đích:** Xử lý và chuẩn hóa kết quả cuối cùng

**Luồng xử lý:**
```
Raw Results → Score Normalization → Result Ranking → Metadata Enrichment → Response Formatting
     ↓              ↓                ↓                ↓                    ↓
Pipeline Output   Score Range      Sort Order      Additional Info     Final Response
                 0.0 - 1.0        by Score        + Context           for User
```

**Code logic (pseudocode):**
```python
def process_final_results(results, top_k_final):
    try:
        # Limit results theo top_k_final
        final_results = results[:top_k_final]
        
        # Normalize scores
        for result in final_results:
            # Ensure scores are in valid range
            result['final_score'] = max(0.0, min(1.0, result['final_score']))
            
            # Add ranking information
            result['final_rank'] = final_results.index(result) + 1
            
            # Add confidence level
            if result['final_score'] >= 0.8:
                result['confidence'] = 'High'
            elif result['final_score'] >= 0.6:
                result['confidence'] = 'Medium'
            else:
                result['confidence'] = 'Low'
        
        # Add summary statistics
        summary = {
            'total_results': len(final_results),
            'avg_score': sum(r['final_score'] for r in final_results) / len(final_results),
            'max_score': max(r['final_score'] for r in final_results),
            'min_score': min(r['final_score'] for r in final_results)
        }
        
        return final_results, summary
        
    except Exception as e:
        logger.error(f"❌ Result processing failed: {e}")
        return results, {}
```

#### 4.5.2 Metadata Enrichment

**Mục đích:** Bổ sung thông tin metadata cho kết quả

**Luồng xử lý:**
```
Basic Results → Parent Law Mapping → Content Enhancement → Metadata Addition → Enriched Results
      ↓              ↓                    ↓                ↓                ↓
Pipeline Output   Law Name Lookup    Content Info      Additional      Complete
                  + Hierarchy        + Structure       Metadata        Results
```

**Code logic (pseudocode):**
```python
def enrich_results_with_metadata(results, pipeline):
    try:
        for result in results:
            aid = result.get('aid')
            
            # Get parent law information
            if hasattr(pipeline.retriever, 'aid_to_parent_law'):
                parent_law = pipeline.retriever.aid_to_parent_law.get(str(aid), 'Unknown')
                result['parent_law_name'] = parent_law
            
            # Add content metadata
            content = result.get('content', '')
            result['content_length'] = len(content)
            result['content_preview'] = content[:200] + '...' if len(content) > 200 else content
            
            # Add processing metadata
            result['processing_timestamp'] = datetime.now().isoformat()
            result['pipeline_version'] = 'v8.3'
        
        return results
        
    except Exception as e:
        logger.warning(f"⚠️ Metadata enrichment failed: {e}")
        return results
```

### 4.6 User Output Stage

#### 4.6.1 Interactive Visualization

**Mục đích:** Hiển thị kết quả với giao diện tương tác

**Luồng xử lý:**
```
Processed Results → Chart Generation → Interactive Display → User Interaction → Enhanced UX
         ↓                ↓                ↓                ↓                ↓
Final Results      Plotly Charts     Streamlit UI      User Clicks      Better
+ Metadata        + Visualizations   + Widgets         + Navigation     Experience
```

**Code logic (pseudocode):**
```python
def create_interactive_visualizations(results):
    # Score comparison chart
    scores_data = {
        'Kết quả': [f"#{i+1}" for i in range(len(results))],
        'Điểm Light Rerank': [r.get('light_reranker_score', 0.0) for r in results],
        'Điểm Retrieval': [r['retrieval_score'] for r in results],
        'Điểm Cross Encoder': [r.get('cross_encoder_score', 0.0) for r in results],
        'Điểm Cuối cùng': [r['final_score'] for r in results]
    }
    
    df_scores = pd.DataFrame(scores_data)
    
    # Create interactive chart
    fig = px.bar(
        df_scores,
        x='Kết quả',
        y=['Điểm Retrieval', 'Điểm Light Rerank', 'Điểm Cross Encoder', 'Điểm Cuối cùng'],
        title='So sánh điểm số các tầng',
        barmode='group'
    )
    
    return fig, df_scores

def display_results_with_tabs(results):
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📄 Xem chi tiết", "📊 So sánh điểm số"])
    
    with tab1:
        # Detailed view
        st.success(f"✅ Tìm thấy {len(results)} kết quả phù hợp!")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"#{i} - {result['aid']} (Điểm: {result['final_score']:.3f})"):
                st.markdown(f"**ID:** {result['aid']}")
                if result.get('parent_law_name'):
                    st.markdown(f"**Tên điều luật:** {result['parent_law_name']}")
                st.markdown("**Nội dung:**")
                st.text(result['content'])
                
                # Score breakdown
                st.markdown("**Điểm từng tầng:**")
                st.markdown(f"- Retrieval: {result['retrieval_score']:.3f}")
                st.markdown(f"- Light Rerank: {result.get('light_reranker_score', 0.0):.3f}")
                st.markdown(f"- Cross Encoder: {result.get('cross_encoder_score', 0.0):.3f}")
    
    with tab2:
        # Score comparison view
        fig, df_scores = create_interactive_visualizations(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Điểm cao nhất", f"{max(df_scores['Điểm Cuối cùng']):.3f}")
        with col2:
            st.metric("Điểm thấp nhất", f"{min(df_scores['Điểm Cuối cùng']):.3f}")
        with col3:
            st.metric("Điểm trung bình", f"{df_scores['Điểm Cuối cùng'].mean():.3f}")
```

#### 4.6.2 Performance Metrics Display

**Mục đích:** Hiển thị metrics hiệu suất cho người dùng

**Luồng xử lý:**
```
Processing Data → Metrics Calculation → Performance Display → User Feedback → Optimization
       ↓                ↓                    ↓                ↓                ↓
Request Info      Timing + Counts      Streamlit UI      User Rating      System
+ Results        + Statistics         + Metrics         + Comments       Improvement
```

**Code logic (pseudocode):**
```python
def display_performance_metrics(start_time, end_time, results_count):
    duration = end_time - start_time
    
    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⏱️ Thời gian xử lý", f"{duration:.2f}s")
    
    with col2:
        st.metric("📊 Kết quả tìm thấy", results_count)
    
    with col3:
        if duration > 0:
            st.metric("⚡ Tốc độ", f"{results_count/duration:.1f} kết quả/s")
    
    # Performance insights
    if duration < 1.0:
        st.success("🚀 Xử lý rất nhanh (< 1 giây)")
    elif duration < 3.0:
        st.info("⚡ Xử lý nhanh (< 3 giây)")
    else:
        st.warning("⏳ Xử lý chậm (> 3 giây) - có thể cần tối ưu")

def create_performance_summary(processing_info):
    summary = {
        'total_queries_processed': processing_info.get('query_count', 0),
        'average_processing_time': processing_info.get('avg_time', 0.0),
        'success_rate': processing_info.get('success_rate', 0.0),
        'cache_hit_rate': processing_info.get('cache_hit_rate', 0.0)
    }
    
    return summary
```

### 4.7 Error Handling và Recovery

#### 4.7.1 Error Handling Strategy

**Mục đích:** Xử lý lỗi một cách graceful và cung cấp feedback hữu ích

**Luồng xử lý:**
```
Error Occurrence → Error Classification → Error Handling → User Feedback → Recovery
       ↓                ↓                    ↓                ↓                ↓
Exception Raised   Error Type         Handle Strategy    User Message     System
                   + Severity         + Fallback        + Guidance       Recovery
```

**Code logic (pseudocode):**
```python
def handle_pipeline_errors(error, stage_name):
    error_handlers = {
        'pipeline_not_ready': handle_pipeline_not_ready,
        'model_loading_failed': handle_model_loading_failed,
        'retrieval_failed': handle_retrieval_failed,
        'reranking_failed': handle_reranking_failed,
        'score_aggregation_failed': handle_score_aggregation_failed
    }
    
    # Classify error
    error_type = classify_error(error)
    
    # Get appropriate handler
    handler = error_handlers.get(error_type, handle_generic_error)
    
    # Handle error
    return handler(error, stage_name)

def handle_pipeline_not_ready(error, stage_name):
    st.error("❌ Pipeline chưa sẵn sàng")
    st.info("💡 Vui lòng kiểm tra:")
    st.info("1. Models đã được tải thành công chưa?")
    st.info("2. FAISS index có tồn tại không?")
    st.info("3. Cấu hình có đúng không?")
    
    # Provide recovery options
    if st.button("🔄 Thử khởi tạo lại Pipeline"):
        st.rerun()

def handle_model_loading_failed(error, stage_name):
    st.error(f"❌ Không thể tải model cho {stage_name}")
    st.info("💡 Nguyên nhân có thể:")
    st.info("1. Model files bị thiếu hoặc hỏng")
    st.info("2. Không đủ memory để load model")
    st.info("3. GPU/CPU compatibility issues")
    
    # Suggest CPU mode
    if st.button("💻 Thử chế độ CPU"):
        st.session_state.force_cpu = True
        st.rerun()
```

#### 4.7.2 Recovery Mechanisms

**Mục đích:** Cung cấp cơ chế khôi phục khi gặp lỗi

**Các cơ chế recovery:**
- **Automatic Retry**: Tự động thử lại với exponential backoff
- **Fallback Modes**: Sử dụng chế độ dự phòng (CPU mode, simplified pipeline)
- **Graceful Degradation**: Giảm chất lượng nhưng vẫn hoạt động
- **User Guidance**: Hướng dẫn người dùng khắc phục

**Code logic (pseudocode):**
```python
def implement_recovery_mechanisms(pipeline, error):
    recovery_strategies = [
        try_force_cpu_mode,
        try_simplified_pipeline,
        try_model_reload,
        try_cache_clear
    ]
    
    for strategy in recovery_strategies:
        try:
            if strategy(pipeline, error):
                logger.info(f"✅ Recovery successful with {strategy.__name__}")
                return True
        except Exception as recovery_error:
            logger.warning(f"⚠️ Recovery strategy {strategy.__name__} failed: {recovery_error}")
    
    return False

def try_force_cpu_mode(pipeline, error):
    """Thử chuyển sang CPU mode"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Reload pipeline với CPU mode
        new_pipeline = LegalQAPipeline()
        if new_pipeline.is_ready:
            st.session_state.pipeline = new_pipeline
            st.session_state.force_cpu = True
            return True
    except Exception as e:
        logger.warning(f"CPU mode recovery failed: {e}")
    
    return False

def try_simplified_pipeline(pipeline, error):
    """Thử sử dụng simplified pipeline (chỉ Tier 1)"""
    try:
        # Disable Tier 2 và Tier 3
        pipeline.use_light_ranking = False
        pipeline.use_cross_encoder = False
        
        # Test với simple query
        test_results = pipeline.predict("test query", top_k_final=1)
        if test_results:
            return True
    except Exception as e:
        logger.warning(f"Simplified pipeline recovery failed: {e}")
    
    return False
```

---

## 5. GIAO DIỆN VÀ BÁO CÁO

### 5.1 Tổng quan giao diện Streamlit

Giao diện của LawBot được xây dựng trên Streamlit với kiến trúc modular và responsive:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│  🎯 MAIN APPLICATION (app.py)                                  │
│  ├── Page Configuration                                        │
│  ├── Sidebar Navigation                                        │
│  ├── Page State Management                                     │
│  └── Error Handling                                            │
├─────────────────────────────────────────────────────────────────┤
│  📱 PAGE MODULES                                               │
│  ├── Search Page (search.py)                                   │
│  ├── Analysis Page (analysis.py)                               │
│  └── System Page (system.py)                                   │
├─────────────────────────────────────────────────────────────────┤
│  🎨 UI COMPONENTS                                              │
│  ├── Interactive Widgets                                       │
│  ├── Data Visualization                                        │
│  ├── Performance Metrics                                       │
│  └── User Feedback                                             │
├─────────────────────────────────────────────────────────────────┤
│  🔧 UTILITY FUNCTIONS                                          │
│  ├── Cache Management                                          │
│  ├── Data Loading                                              │
│  ├── Error Handling                                            │
│  └── Performance Monitoring                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Main Application Structure

#### 5.2.1 App Configuration và Setup

**Mục đích:** Cấu hình chính cho ứng dụng Streamlit

**Luồng khởi tạo:**
```
App Startup → Configuration Load → Page Setup → Navigation Setup → Ready for User
      ↓              ↓                ↓              ↓                ↓
Streamlit Run    Config Files     Page Config     Sidebar Nav      App Ready
                 + Settings       + Layout        + Routing        for Input
```

**Code logic (pseudocode):**
```python
def render_app():
    # Page configuration
    st.set_page_config(
        page_title="Hệ thống Hỏi-Đáp Pháp luật",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide default navigation
    hide_default_navigation = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebarNav"] {display: none !important;}
    </style>
    """
    st.markdown(hide_default_navigation, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🎯 Hệ thống QA Pháp luật")
    
    # Page options
    page_options = {
        "🔍 Tìm kiếm & Hỏi đáp": search.render_search_page,
        "📊 Phân tích & Báo cáo": analysis.render_analysis_page,
        "🔧 Trạng thái": system.main
    }
    
    # Page selection
    selected_page = st.sidebar.selectbox(
        "Chọn trang:",
        list(page_options.keys()),
        index=0
    )
    
    # Render selected page
    if selected_page in page_options:
        page_options[selected_page]()
```

#### 5.2.2 Page State Management

**Mục đích:** Quản lý trạng thái giữa các trang và sessions

**Luồng quản lý state:**
```
Page Switch → State Cleanup → New Page Load → State Initialization → Page Ready
      ↓            ↓              ↓              ↓                ↓
User Click    Clear Old      Load New      Initialize New     Page Active
on Nav        Page State     Page          Page State        for User
```

**Code logic (pseudocode):**
```python
def manage_page_state(selected_page_title):
    # Initialize current page in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = selected_page_title
    
    # Clear page-specific states when switching pages
    if st.session_state.current_page != selected_page_title:
        # Clear all page-specific states
        for key in list(st.session_state.keys()):
            if key.endswith("_page_loaded"):
                del st.session_state[key]
        
        # Update current page
        st.session_state.current_page = selected_page_title
        st.rerun()
    
    # Mark page as loaded
    page_key = f"{selected_page_title.lower().replace(' ', '_')}_page_loaded"
    if page_key not in st.session_state:
        st.session_state[page_key] = True
```

### 5.3 Search Page (search.py)

#### 5.3.1 Search Interface Design

**Mục đích:** Cung cấp giao diện tìm kiếm chính cho người dùng

**Luồng xử lý:**
```
User Input → Query Processing → Search Execution → Results Display → User Interaction
      ↓            ↓                ↓                ↓                ↓
Question Text   Text Clean      Pipeline Run     Results Show     User Clicks
+ Parameters   + Validation    + Processing      + Charts         + Navigation
```

**Code logic (pseudocode):**
```python
def render_search_page():
    # Clean page state
    if "search_page_loaded" not in st.session_state:
        st.session_state.search_page_loaded = True
    
    # Page title và description
    st.title("🔍 Legal QA - Hệ thống Hỏi đáp Pháp luật")
    st.markdown("Hệ thống AI hỗ trợ tìm kiếm và trả lời câu hỏi pháp luật Việt Nam")
    
    # Sidebar configuration
    with st.sidebar.expander("⚙️ Cấu hình tìm kiếm", expanded=True):
        final_results_count = st.slider(
            "Số lượng kết quả cuối cùng",
            min_value=1, max_value=50, value=3
        )
        
        search_aggressiveness = st.selectbox(
            "Mức độ tìm kiếm",
            options=["conservative", "balanced", "aggressive"],
            index=1
        )
        
        force_cpu = st.checkbox("Sử dụng CPU", value=False)
    
    # Random question feature
    training_questions, test_questions = load_random_questions()
    if training_questions or test_questions:
        with st.sidebar.expander("🎲 Câu hỏi ngẫu nhiên", expanded=True):
            question_source = st.selectbox(
                "Chọn nguồn câu hỏi",
                options=["Training Dataset", "Test Dataset", "Cả hai"]
            )
            
            if st.button("🎲 Lấy câu hỏi ngẫu nhiên"):
                random_q = get_random_question(
                    training_questions + test_questions, "combined"
                )
                st.session_state.random_question = random_q
                st.session_state.current_question = random_q
```

#### 5.3.2 Search Execution và Results Display

**Mục đích:** Thực hiện tìm kiếm và hiển thị kết quả

**Luồng xử lý:**
```
Search Button → Pipeline Execution → Results Processing → Display → User Feedback
      ↓              ↓                ↓                ↓            ↓
User Click     3-Tier Pipeline    Score Aggreg    Interactive    User Rating
               + Processing       + Ranking       Visualization   + Comments
```

**Code logic (pseudocode):**
```python
def execute_search(query, final_results_count, search_aggressiveness):
    # Calculate optimal parameters
    params = calculate_optimal_parameters(final_results_count, search_aggressiveness)
    
    # Display calculated parameters
    with st.expander("📊 Thông số tìm kiếm được tính toán tự động"):
        st.markdown(f"**🎯 Tầng 1 - Retrieval:** {params['top_k_retrieval']} ứng viên")
        st.markdown(f"**⚡ Tầng 2 - Light Reranking:** {params['top_k_light_reranking']} ứng viên")
        st.markdown(f"**🎯 Tầng 3 - Final Reranking:** {params['top_k_final']} kết quả cuối cùng")
    
    # Performance monitoring
    start_time = time.time()
    
    with st.spinner("🔎 Đang tìm kiếm... vui lòng đợi"):
        # Get results from pipeline
        results = pipeline.predict(
            query,
            top_k_retrieval=params["top_k_retrieval"],
            top_k_final=params["top_k_final"]
        )
    
    end_time = time.time()
    
    # Display performance metrics
    display_performance_metrics(start_time, end_time, len(results))
    
    # Display results in tabs
    tab1, tab2 = st.tabs(["📄 Xem chi tiết", "📊 So sánh điểm số"])
    
    with tab1:
        display_detailed_results(results)
    
    with tab2:
        display_score_comparison(results)
```

#### 5.3.3 Results Visualization

**Mục đích:** Hiển thị kết quả với biểu đồ tương tác

**Luồng xử lý:**
```
Search Results → Data Processing → Chart Generation → Interactive Display → User Analysis
       ↓              ↓                ↓                ↓                ↓
Pipeline Output   Data Format      Plotly Charts     Streamlit UI      User
+ Metadata       + Aggregation    + Visualizations  + Widgets         Interaction
```

**Code logic (pseudocode):**
```python
def display_score_comparison(results):
    # Score comparison chart
    scores_data = {
        'Kết quả': [f"#{i+1}" for i in range(len(results))],
        'Điểm Light Rerank': [r.get("light_reranker_score", 0.0) for r in results],
        'Điểm Retrieval': [r["retrieval_score"] for r in results],
        'Điểm Cross Encoder': [r.get("cross_encoder_score", 0.0) for r in results]
    }
    
    df_scores = pd.DataFrame(scores_data)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Điểm cao nhất", f"{max(scores_data['Điểm Light Rerank']):.3f}")
    with col2:
        st.metric("Điểm thấp nhất", f"{min(scores_data['Điểm Light Rerank']):.3f}")
    with col3:
        st.metric("Điểm trung bình", f"{sum(scores_data['Điểm Light Rerank'])/len(scores_data['Điểm Light Rerank']):.3f}")
    
    # Interactive chart
    st.bar_chart(df_scores.set_index("Kết quả"))

def display_detailed_results(results):
    st.success(f"✅ Tìm thấy {len(results)} kết quả phù hợp!")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"#{i} - {result['aid']} (Điểm: {result['final_score']:.3f})", expanded=(i <= 3)):
            st.markdown(f"**ID:** {result['aid']}")
            
            if result.get("parent_law_name"):
                st.markdown(f"**Tên điều luật:** {result['parent_law_name']}")
            
            st.markdown("**Nội dung:**")
            st.text(result["content"])
            
            # Score breakdown
            st.markdown(f"**Điểm retrieval:** {result['retrieval_score']:.3f}")
            st.markdown(f"**Điểm light rerank:** {result.get('light_reranker_score', 0.0):.3f}")
            st.markdown(f"**Điểm cross encoder:** {result.get('cross_encoder_score', 0.0):.3f}")
```

### 5.4 Analysis Page (analysis.py)

#### 5.4.1 Comprehensive Evaluation Interface

**Mục đích:** Cung cấp giao diện đánh giá toàn diện hệ thống

**Luồng xử lý:**
```
Page Load → Auto-evaluation → Results Processing → Visualization → User Analysis
     ↓            ↓                ↓                ↓            ↓
Streamlit UI   Pipeline Eval    Metrics Calc     Charts       User
+ Navigation   + Metrics        + Aggregation    + Tables     Interaction
```

**Code logic (pseudocode):**
```python
def render_analysis_page():
    # Clean page state
    if "analysis_page_loaded" not in st.session_state:
        st.session_state.analysis_page_loaded = True
        st.session_state.comprehensive_eval_results = None
        st.session_state.eval_loading = False
    
    st.title("📊 Báo cáo phân tích chi tiết")
    st.markdown("Phân tích toàn diện hiệu suất từng tầng và khuyến nghị cải thiện")
    
    # Load analysis data
    analysis_data = load_analysis_data()
    model_status = analysis_data.get("models", {})
    faiss_status = analysis_data.get("faiss", {})
    eval_data = analysis_data.get("evaluation")
    
    # Create tabs
    tab1, tab2 = st.tabs([
        "🏥 Tổng quan",
        "🧪 Phân tích đánh giá toàn diện"
    ])
    
    with tab1:
        render_overview_tab(model_status, faiss_status)
    
    with tab2:
        render_comprehensive_evaluation_tab()
```

#### 5.4.2 Auto-evaluation System

**Mục đích:** Tự động chạy evaluation và cache kết quả

**Luồng xử lý:**
```
Auto-trigger → Pipeline Check → Evaluation Run → Results Cache → Display
      ↓            ↓                ↓              ↓            ↓
Page Load      Pipeline Ready   Metrics Calc    Save Results   Show User
+ Timer        + Validation     + Processing    + Cache        + Charts
```

**Code logic (pseudocode):**
```python
def auto_run_comprehensive_evaluation():
    try:
        logger.info("🔄 Auto-running comprehensive evaluation...")
        
        # First, try to load from saved file
        cached_results = load_latest_comprehensive_evaluation()
        if cached_results:
            logger.info("✅ Loaded comprehensive evaluation results from cache file")
            return cached_results
        
        logger.info("🔄 No cached results found, running fresh evaluation...")
        
        # Get pipeline
        pipeline = get_pipeline_lazy()
        if not pipeline:
            logger.warning("⚠️ Pipeline not available for auto-evaluation")
            return None
        
        # Run evaluation
        eval_results = run_comprehensive_evaluation(pipeline)
        
        if eval_results:
            logger.info("✅ Auto-comprehensive evaluation completed successfully")
            return eval_results
        else:
            logger.warning("⚠️ Auto-comprehensive evaluation returned no results")
            return None
            
    except Exception as e:
        logger.error(f"❌ Auto-comprehensive evaluation failed: {e}")
        return None

def run_comprehensive_evaluation(pipeline, test_queries=None):
    if test_queries is None:
        # Load validation sets
        validation_dir = Path("features/validation_sets")
        if validation_dir.exists():
            manager = ValidationSetManager()
            tier1_val, tier2_val, tier3_val = manager.load_validation_sets(str(validation_dir))
            
            if tier1_val and tier2_val and tier3_val:
                test_queries = [
                    item.get("query", "") for item in tier1_val[:10]
                    if item.get("query")
                ]
    
    # Initialize results structure
    evaluation_results = {
        "tier_1": {}, "tier_2": {}, "tier_3": {}, "combined": {}
    }
    
    # Process queries in batches
    for query in test_queries:
        results = pipeline.predict(query, top_k_final=10)
        if results:
            tier_metrics = calculate_tier_metrics_optimized(results, [1, 3, 5, 10])
            
            # Update evaluation results
            for tier, metrics in tier_metrics.items():
                for metric_name, values in metrics.items():
                    if metric_name not in evaluation_results[tier]:
                        evaluation_results[tier][metric_name] = []
                    evaluation_results[tier][metric_name].extend(values)
    
    # Calculate final averages
    final_results = calculate_final_averages_optimized(evaluation_results)
    
    return final_results
```

#### 5.4.3 Performance Metrics Visualization

**Mục đích:** Hiển thị metrics hiệu suất với biểu đồ tương tác

**Luồng xử lý:**
```
Evaluation Results → Metrics Processing → Chart Generation → Interactive Display → User Analysis
         ↓                ↓                ↓                ↓                ↓
Raw Metrics      Data Format      Plotly Charts     Streamlit UI      User
+ Calculations   + Aggregation    + Visualizations  + Widgets         Interaction
```

**Code logic (pseudocode):**
```python
def create_tier_performance_chart(eval_results):
    if not eval_results:
        return None
    
    # Prepare data for visualization
    metrics = ["precision", "recall", "f1", "ndcg", "mrr", "quality"]
    tiers = ["tier_1", "tier_2", "tier_3", "combined"]
    tier_names = ["Retrieval", "Light Reranker", "Cross Encoder", "Combined"]
    
    # Create data for the chart
    chart_data = []
    for i, tier in enumerate(tiers):
        for metric in metrics:
            avg_value = eval_results[tier].get(f"{metric}_avg", 0.0)
            chart_data.append({
                "Tier": tier_names[i],
                "Metric": metric.upper(),
                "Value": avg_value,
                "Tier_Type": "Individual" if i < 3 else "Combined",
                "Color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i]
            })
    
    df = pd.DataFrame(chart_data)
    
    # Create enhanced bar chart
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Tier",
        title="🎯 Tier Performance Comparison - Comprehensive Metrics",
        color_discrete_map={
            "Retrieval": "#1f77b4",
            "Light Reranker": "#ff7f0e",
            "Cross Encoder": "#2ca02c",
            "Combined": "#d62728"
        },
        barmode="group",
        text="Value"
    )
    
    # Enhance chart appearance
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont_size=10
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="📊 Metrics",
        yaxis_title="📈 Score",
        showlegend=True,
        title_font_size=16,
        title_x=0.5,
        plot_bgcolor="white",
        bargap=0.2,
        bargroupgap=0.1
    )
    
    return fig

def create_tier_improvement_chart(eval_results):
    if not eval_results:
        return None
    
    # Calculate improvement percentages
    improvement_data = []
    
    for metric in ["precision", "recall", "f1", "ndcg", "mrr", "quality"]:
        # Get best individual tier performance
        individual_scores = [
            eval_results["tier_1"].get(f"{metric}_avg", 0.0),
            eval_results["tier_2"].get(f"{metric}_avg", 0.0),
            eval_results["tier_3"].get(f"{metric}_avg", 0.0)
        ]
        best_individual = max(individual_scores)
        combined_score = eval_results["combined"].get(f"{metric}_avg", 0.0)
        
        if best_individual > 0:
            improvement = ((combined_score - best_individual) / best_individual) * 100
        else:
            improvement = 0.0
        
        improvement_data.append({
            "Metric": metric.upper(),
            "Best Individual": best_individual,
            "Combined": combined_score,
            "Improvement %": improvement
        })
    
    df = pd.DataFrame(improvement_data)
    
    # Create enhanced improvement chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="🏆 Best Individual Tier",
        x=df["Metric"],
        y=df["Best Individual"],
        marker_color="lightblue",
        text=df["Best Individual"].round(3),
        textposition="outside"
    ))
    
    fig.add_trace(go.Bar(
        name="🚀 Combined Pipeline",
        x=df["Metric"],
        y=df["Combined"],
        marker_color="darkblue",
        text=df["Combined"].round(3),
        textposition="outside"
    ))
    
    # Add improvement annotations
    for i, row in df.iterrows():
        if row["Improvement %"] > 0:
            fig.add_annotation(
                x=row["Metric"],
                y=row["Combined"] + 0.01,
                text=f"+{row['Improvement %']:.1f}%",
                showarrow=False,
                font=dict(color="green", size=12, weight="bold"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1
            )
    
    fig.update_layout(
        title="📈 Performance Improvement: Individual vs Combined Pipeline",
        height=500,
        barmode="group",
        xaxis_title="📊 Metrics",
        yaxis_title="📈 Score",
        plot_bgcolor="white",
        showlegend=True
    )
    
    return fig
```

### 5.5 System Page (system.py)

#### 5.5.1 System Health Dashboard

**Mục đích:** Hiển thị trạng thái tổng quan của hệ thống

**Luồng xử lý:**
```
Page Load → System Check → Status Collection → Health Calculation → Dashboard Display
     ↓            ↓              ↓                ↓                ↓
Streamlit UI   Pipeline      Model Status     Health Score     Interactive
+ Navigation   + Models      + FAISS Index    + Metrics        Dashboard
```

**Code logic (pseudocode):**
```python
def create_system_health_dashboard(system_data):
    st.subheader("🏥 System Health Dashboard")
    
    if not system_data:
        st.warning("⚠️ Không thể load dữ liệu hệ thống")
        return
    
    # Overall system status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check if models are ready
        model_status = system_data.get("model_status", {})
        models_ready = sum(
            1 for model in model_status.values() 
            if model.get("status") == "ready"
        )
        total_models = len(model_status)
        
        if total_models > 0:
            health_percentage = (models_ready / total_models) * 100
        else:
            health_percentage = 0
        
        st.metric("System Health", f"{health_percentage:.0f}%")
    
    with col2:
        # FAISS Index status
        faiss_status = system_data.get("faiss_status", {})
        faiss_ready = "✅ Ready" if faiss_status.get("status") == "ready" else "❌ Not Ready"
        st.metric("FAISS Index", faiss_ready)
    
    with col3:
        # Last update time
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    # Health gauge chart
    if system_data.get("evaluation_data"):
        health_score = system_data["evaluation_data"].get(
            "pipeline_health_score", health_percentage
        )
    else:
        health_score = health_percentage
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "System Health Score"},
        delta={"reference": 100},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightcoral"},
                {"range": [50, 80], "color": "lightyellow"},
                {"range": [80, 100], "color": "lightgreen"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
```

#### 5.5.2 Tier Configuration Information

**Mục đích:** Hiển thị thông tin cấu hình chi tiết từng tầng

**Luồng xử lý:**
```
System Data → Tier Info Extraction → Configuration Display → Status Check → User Info
      ↓              ↓                    ↓                ↓            ↓
Raw System      Tier-specific      Config Tables      Health Check    Detailed
Data            Config Data        + Status           + Validation     Information
```

**Code logic (pseudocode):**
```python
def create_tier_configuration_info(system_data):
    st.subheader("🎯 Thông tin Cấu hình từng Tầng")
    
    if not system_data:
        st.warning("⚠️ Không thể load thông tin cấu hình")
        return
    
    model_status = system_data.get("model_status", {})
    
    # Create tier configuration table
    tier_configs = {
        "Bi-Encoder (Tầng 1)": {
            "model": model_status.get("bi_encoder", {}),
            "description": "Retrieval model sử dụng Sentence Transformers",
            "status_key": "bi_encoder"
        },
        "Light Reranker (Tầng 2)": {
            "model": model_status.get("light_reranker", {}),
            "description": "Light reranking model sử dụng PhoBERT",
            "status_key": "light_reranker"
        },
        "Cross-Encoder (Tầng 3)": {
            "model": model_status.get("cross_encoder", {}),
            "description": "Cross-encoder model kết hợp ensemble",
            "status_key": "cross_encoder"
        }
    }
    
    # Display tier information
    for tier_name, config in tier_configs.items():
        with st.expander(f"🔧 {tier_name}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Mô tả:** {config['description']}")
                
                model_info = config["model"]
                if model_info:
                    status = model_info.get("status", "unknown")
                    model_path = model_info.get("path", "N/A")
                    size_mb = model_info.get("size_mb", 0)
                    
                    # Status indicator
                    if status == "ready":
                        st.success(f"✅ Status: Ready")
                    elif status == "partially_ready":
                        st.warning(f"⚠️ Status: Partially Ready")
                    else:
                        st.error(f"❌ Status: {status}")
                    
                    st.write(f"**Model Path:** `{model_path}`")
                    st.write(f"**Model Size:** {size_mb:.1f} MB")
                else:
                    st.error("❌ Không có thông tin model")
            
            with col2:
                # Quick status check
                if model_info and model_info.get("status") == "ready":
                    st.success("✅ Hoạt động bình thường")
                else:
                    st.warning("⚠️ Cần kiểm tra")
```

### 5.6 Reporting và Analytics System

#### 5.6.1 Comprehensive Evaluation Reports

**Mục đích:** Tạo báo cáo đánh giá toàn diện hệ thống

**Luồng xử lý:**
```
Evaluation Run → Metrics Calculation → Report Generation → File Storage → User Access
       ↓              ↓                ↓                ↓            ↓
Pipeline Test   Performance      Report Format     Save to Disk    Display
+ Validation    Metrics         + Visualizations  + Metadata      + Download
```

**Code logic (pseudocode):**
```python
def save_comprehensive_evaluation_results(results, filename=None):
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_evaluation_{timestamp}.json"
        
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        # Prepare data for saving
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "metadata": {
                "version": "v8.3",
                "type": "comprehensive_evaluation",
                "source": "analysis_page"
            }
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Comprehensive evaluation results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"❌ Failed to save evaluation results: {e}")
        return None

def load_latest_comprehensive_evaluation():
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return None
        
        # Find the latest comprehensive evaluation file
        comp_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
        if not comp_files:
            return None
        
        # Get the most recent file
        latest_file = max(comp_files, key=lambda p: p.stat().st_mtime)
        
        # Check if file is recent (within last 24 hours)
        file_age = time.time() - latest_file.stat().st_mtime
        if file_age > 86400:  # 24 hours in seconds
            logger.info("⚠️ Latest evaluation file is older than 24 hours")
            return None
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded evaluation results from: {latest_file}")
        return data.get("results")
        
    except Exception as e:
        logger.error(f"❌ Failed to load evaluation results: {e}")
        return None
```

#### 5.6.2 Performance Metrics và Insights

**Mục đích:** Cung cấp insights về hiệu suất hệ thống

**Luồng xử lý:**
```
Raw Metrics → Analysis → Insights Generation → Recommendations → User Display
      ↓          ↓            ↓                ↓                ↓
Performance   Statistical    Performance      Improvement      Actionable
Data          Analysis       Insights         Suggestions      Information
```

**Code logic (pseudocode):**
```python
def create_performance_insights(eval_results):
    st.subheader("💡 Phân tích Chi tiết & Khuyến nghị")
    
    # Analyze each tier's performance
    tier_analysis = {}
    for tier in ["tier_1", "tier_2", "tier_3"]:
        tier_metrics = eval_results[tier]
        avg_f1 = tier_metrics.get("f1_avg", 0.0)
        avg_quality = tier_metrics.get("quality_avg", 0.0)
        
        if avg_f1 >= 0.8 and avg_quality >= 0.7:
            status = "✅ Xuất sắc"
            recommendation = "Tầng này hoạt động rất tốt, không cần cải thiện"
        elif avg_f1 >= 0.6 and avg_quality >= 0.5:
            status = "🟡 Tốt"
            recommendation = "Có thể cải thiện thêm để đạt hiệu suất cao hơn"
        else:
            status = "🔴 Cần cải thiện"
            recommendation = "Cần xem xét lại training data và hyperparameters"
        
        tier_analysis[tier] = {
            "status": status,
            "f1_score": avg_f1,
            "quality_score": avg_quality,
            "recommendation": recommendation
        }
    
    # Display tier analysis
    col1, col2, col3 = st.columns(3)
    tier_names = {
        "tier_1": "🎯 Tầng 1 (Retrieval)",
        "tier_2": "⚡ Tầng 2 (Light Reranker)",
        "tier_3": "🎯 Tầng 3 (Cross Encoder)"
    }
    
    for i, (tier, analysis) in enumerate(tier_analysis.items()):
        with [col1, col2, col3][i]:
            st.subheader(tier_names[tier])
            st.metric("Status", analysis["status"])
            st.metric("F1 Score", f"{analysis['f1_score']:.3f}")
            st.metric("Quality", f"{analysis['quality_score']:.3f}")
            st.info(analysis["recommendation"])
```

---

## 6. KỸ THUẬT CODE VÀ KIẾN TRÚC

### 6.1 Centralized Configuration Management

**Mục đích:** Tập trung hóa cấu hình để đảm bảo tính nhất quán và dễ bảo trì

#### 6.1.1 Model Configuration (`config/models.py`)

**MODEL_TYPES Dictionary:**
```python
MODEL_TYPES = {
    "bi_encoder": {
        "name": "bi_encoder",
        "display_name": "Bi-Encoder",
        "directory_prefix": "bi_encoder",
        "purpose": "Retrieval",
        "tier": 1,
        "performance": "Fast retrieval (~1000+ docs/sec)",
        "techniques": ["Contrastive Learning", "HNM", "ADAPT"]
    },
    "light_reranker": {
        "name": "light_reranker", 
        "display_name": "Light Reranker",
        "directory_prefix": "light_reranker",
        "purpose": "Light filtering",
        "tier": 2,
        "performance": "Fast filtering (~500+ docs/sec)",
        "techniques": ["PhoBERT", "ADAPT", "HPO"]
    },
    "cross_encoder": {
        "name": "cross_encoder",
        "display_name": "Cross-Encoder",
        "directory_prefix": "cross_encoder", 
        "purpose": "Final ranking",
        "tier": 3,
        "performance": "High accuracy (~100+ docs/sec)",
        "techniques": ["Ensemble", "HPO", "HNM"]
    }
}
```

**MODEL_DIRECTORY_MAPPING:**
```python
MODEL_DIRECTORY_MAPPING = {
    "bi_encoder": "models/bi_encoder",
    "light_reranker": "models/light_reranker", 
    "cross_encoder": "models/cross_encoder"
}
```

**MODEL_STATUS_KEYS:**
```python
MODEL_STATUS_KEYS = {
    "bi_encoder": "bi_encoder",
    "light_reranker": "light_reranker",
    "cross_encoder": "cross_encoder"
}
```

#### 6.1.2 Path Configuration (`config/paths.py`)

**TRAINING_DATA_PATHS:**
```python
TRAINING_DATA_PATHS = {
    "bi_encoder": {
        "primary": "features/processed_data_{timestamp}/bi_encoder_train.jsonl",
        "alternative": "features/processed_data/bi_encoder_train.jsonl"
    },
    "light_reranker": {
        "primary": "features/processed_data_{timestamp}/training_data.jsonl",
        "alternative": "features/processed_data/training_data.jsonl"
    },
    "cross_encoder": {
        "primary": "features/processed_data_{timestamp}/training_data.jsonl", 
        "alternative": "features/processed_data/training_data.jsonl"
    }
}
```

**DATA_QUALITY_THRESHOLDS:**
```python
DATA_QUALITY_THRESHOLDS = {
    "bi_encoder": {
        "min_samples": 1000,
        "min_avg_length": 50,
        "required_fields": ["question", "answer", "context"]
    },
    "light_reranker": {
        "min_samples": 500,
        "min_avg_length": 30,
        "required_fields": ["question", "answer", "negative"]
    },
    "cross_encoder": {
        "min_samples": 300,
        "min_avg_length": 25,
        "required_fields": ["question", "answer", "label"]
    }
}
```

**Helper Functions:**
```python
def validate_training_data_paths():
    """Validate all training data paths and return validation results
    Returns: {
        'tier_1': {'status': 'ready', 'data_source': 'real', 'file_count': int, 'files': [str]},
        'tier_2': {'status': 'ready', 'data_source': 'real', 'file_count': int, 'files': [str]},
        'tier_3': {'status': 'ready', 'data_source': 'real', 'file_count': int, 'files': [str]},
        'overall': {'status': 'ready', 'real_data_available': bool, 'tiers_ready': int},
        'latest_data_dir': str
    }
    """
    
def get_training_data_path(tier_name: str) -> str:
    """Get the correct training data path for a specific tier"""
    
def get_latest_processed_data_dir() -> str:
    """Find the latest timestamped processed data directory"""
```

#### 6.1.3 Configuration Integration

**Workflow Integration:**
```python
# In run_workflow.py
from config.paths import validate_training_data_paths, get_training_data_path
from config.models import get_model_config, get_display_name

# Validate data before training
validation_results = validate_training_data_paths()
if validation_results["overall"]["status"] != "ready":
    logger.error("❌ Training data validation failed")
    return False

# Get model-specific paths
bi_encoder_path = get_training_data_path("bi_encoder")
light_reranker_path = get_training_data_path("light_reranker")
cross_encoder_path = get_training_data_path("cross_encoder")
```

**UI Integration:**
```python
# In app/pages/system.py and app/pages/analysis.py
from config.models import MODEL_STATUS_KEYS, get_display_name

# Use centralized keys
tier_configs = {
    "Bi-Encoder (Tầng 1)": {
        "model": model_status.get(MODEL_STATUS_KEYS["bi_encoder"], {}),
        "status_key": MODEL_STATUS_KEYS["bi_encoder"],
    },
    # ... other tiers
}
```

### 6.2 Kiến trúc Code tổng thể

#### 6.2.1 Design Patterns và Principles

**Mục đích:** Áp dụng các design patterns và principles để tạo code maintainable và scalable

**Các patterns được sử dụng:**
- **Factory Pattern**: Model creation và pipeline initialization
- **Strategy Pattern**: Different reranking strategies
- **Observer Pattern**: Progress tracking và monitoring
- **Template Method Pattern**: Base training scripts
- **Singleton Pattern**: Configuration management

**Code structure:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    CODE ARCHITECTURE PATTERNS                   │
├─────────────────────────────────────────────────────────────────┤
│  🏭 FACTORY PATTERN                                            │
│  ├── ModelFactory: Tạo models theo configuration               │
│  ├── PipelineFactory: Khởi tạo pipeline với models            │
│  └── DatasetFactory: Tạo datasets theo loại training          │
├─────────────────────────────────────────────────────────────────┤
│  🎯 STRATEGY PATTERN                                           │
│  ├── RetrievalStrategy: Different retrieval methods            │
│  ├── RerankingStrategy: Different reranking approaches        │
│  └── EvaluationStrategy: Different evaluation metrics          │
├─────────────────────────────────────────────────────────────────┤
│  👁️ OBSERVER PATTERN                                           │
│  ├── ProgressTracker: Track training progress                  │
│  ├── PipelineMonitor: Monitor pipeline execution               │
│  └── MetricsCollector: Collect performance metrics             │
├─────────────────────────────────────────────────────────────────┤
│  📋 TEMPLATE METHOD PATTERN                                    │
│  ├── BaseTrainingScript: Template cho training scripts         │
│  ├── BaseEvaluator: Template cho evaluation                   │
│  └── BaseTransforms: Template cho data transforms             │
└─────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Module Organization và Dependency Management

**Mục đích:** Tổ chức code theo modules với dependency management rõ ràng

**Cấu trúc dependencies:**
```
app/ (UI Layer)
├── app.py (Main app)
├── pages/ (Page modules)
│   ├── search.py
│   ├── analysis.py
│   └── system.py
└── Dependencies: streamlit, plotly, pandas

core/ (Business Logic Layer)
├── pipeline.py (Pipeline orchestrator)
├── retrieval.py (Retrieval engine)
├── reranking.py (Reranking engine)
├── datasets/ (Dataset classes)
├── transforms/ (Data transforms)
├── utils/ (Utility functions)
└── Dependencies: torch, transformers, faiss

config/ (Configuration Layer)
├── default.yml (Default config)
├── loader.py (Config loader)
├── schemas.py (Pydantic schemas)
└── Dependencies: pydantic, pyyaml

training/ (Training Layer)
├── engine.py (Training engine)
├── base_script.py (Base training)
├── run_*.py (Training scripts)
└── Dependencies: torch, transformers, optuna
```

### 6.3 Core Pipeline Architecture

#### 6.3.1 Pipeline Orchestrator Design

**Mục đích:** Thiết kế pipeline orchestrator để điều phối 3 tiers

**Kiến trúc:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATOR                        │
├─────────────────────────────────────────────────────────────────┤
│  🎯 LegalQAPipeline Class                                      │
│  ├── __init__(): Model loading và initialization               │
│  ├── predict(): Main prediction method                         │
│  ├── get_pipeline_status(): Health check                       │
│  └── cleanup(): Resource cleanup                               │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ Component Management                                        │
│  ├── retriever: RetrievalEngine instance                       │
│  ├── reranker: RerankingEngine instance                        │
│  ├── config: Configuration management                          │
│  └── state: Pipeline state management                          │
├─────────────────────────────────────────────────────────────────┤
│  🔄 Execution Flow Control                                      │
│  ├── Tier 1: Bi-encoder retrieval                              │
│  ├── Tier 2: Light reranking                                   │
│  ├── Tier 3: Cross-encoder reranking                           │
│  └── Score aggregation và final ranking                        │
└─────────────────────────────────────────────────────────────────┘
```

**Code logic (pseudocode):**
```python
class LegalQAPipeline:
    def __init__(self, bi_encoder_path=None, reranker_paths=None):
        self.is_ready = False
        self.loaded_model_paths = {}
        
        try:
            # Validate parent law mapping
            ensure_parent_law_mapping()
            
            # Load Retriever (Tier 1)
            self.retriever = RetrievalEngine(
                bi_encoder_path=bi_encoder_path or self._get_latest_bi_encoder(),
                faiss_index_path=self._get_faiss_index_path(),
                content_map_path=self._get_content_map_path(),
                index_to_aid_path=self._get_index_to_aid_path()
            )
            
            # Load Reranker (Tier 2 & 3)
            reranker_configs = self._resolve_reranker_paths(reranker_paths)
            if reranker_configs:
                self.reranker = RerankingEngine(reranker_configs)
            else:
                self.reranker = None
            
            self.is_ready = True
            
        except Exception as e:
            logger.error(f"Failed to initialize LegalQAPipeline: {e}")
            self.is_ready = False
            raise
    
    def predict(self, query, top_k_retrieval=None, top_k_final=None):
        """Main prediction method với 3-tier architecture"""
        if not self.is_ready:
            raise RuntimeError("Pipeline is not ready")
        
        try:
            # Tier 1: Bi-Encoder Retrieval
            documents = self.retriever.retrieve(query, top_k=top_k_retrieval)
            
            # Initialize scores
            for doc in documents:
                doc['light_reranker_score'] = 0.0
                doc['cross_encoder_score'] = 0.0
            
            # Tier 2: Light Reranking
            if self.reranker and self.reranker.is_ready:
                documents = self.reranker.rank_light(query, documents)
            
            # Tier 3: Cross-Encoder Reranking
            if self.reranker and self.reranker.is_ready:
                documents = self.reranker.rank_cross(query, documents)
            
            # Score aggregation và final ranking
            documents = self._combine_scores(documents)
            final_results = sorted(
                documents, 
                key=lambda x: x['final_score'], 
                reverse=True
            )[:top_k_final]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
```

#### 6.2.2 Component Interface Design

**Mục đích:** Thiết kế interfaces nhất quán cho các components

**Interface contracts:**
```python
# Base interface cho retrieval components
class RetrievalInterface(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass

# Base interface cho reranking components
class RerankingInterface(ABC):
    @abstractmethod
    def rank_light(self, query: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def rank_cross(self, query: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        pass
```

### 6.3 Configuration Management

#### 6.3.1 Centralized Path Configuration với Automated Validation

**Mục đích:** Quản lý cấu hình tập trung với automated path discovery và validation

**Centralized Path Configuration:**
```python
# config/paths.py - Centralized path management
TRAINING_DATA_PATHS = {
    "primary": {
        "bi_encoder": Path("features/processed_data/bi_encoder_train.jsonl"),
        "cross_encoder": Path("features/processed_data/cross_encoder_train.jsonl"),
        "processed_corpus": Path("features/processed_data/processed_corpus.json"),
        "training_data": Path("features/processed_data/training_data.jsonl"),
        "negative_pool": Path("features/processed_data/negative_pool.jsonl"),
    },
    "validation": {
        "tier_1": Path("features/validation_sets/tier_1_validation.jsonl"),
        "tier_2": Path("features/validation_sets/tier_2_validation.jsonl"),
        "tier_3": Path("features/validation_sets/tier_3_validation.jsonl"),
    }
}

# Automated data freshness validation
def validate_training_data_paths() -> dict:
    """Automatically find latest processed data directory and validate"""
    features_dir = Path("features")
    processed_dirs = [d for d in features_dir.iterdir() 
                     if d.is_dir() and d.name.startswith("processed_data_")]
    
    if processed_dirs:
        latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
        return {"latest_data_dir": str(latest_dir), "status": "ready"}
    return {"status": "missing_data"}

def get_training_data_path(tier: str, data_type: str) -> Path:
    """Get training data path with automatic latest directory discovery"""
    processed_dirs = [d for d in Path("features").iterdir() 
                     if d.is_dir() and d.name.startswith("processed_data_")]
    
    if processed_dirs:
        latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
        return latest_dir / f"{data_type}.jsonl"
    
    # Fallback to primary paths
    return TRAINING_DATA_PATHS["primary"].get(data_type, Path(""))
```

**YAML Configuration với Pydantic Validation:**
```yaml
# config/default.yml
paths:
  data_dir: "data"
  model_dir: "models"
  reports_dir: "reports"

bi_encoder:
  model_name: "bkai-foundation-models/vietnamese-bi-encoder"
  enhancement_techniques:
    contrastive_learning:
      enabled: true
      loss_type: "TripletLoss"
      max_length: 256
      batch_size: 16
      epochs: 5

reranker_pipeline:
  light_reranker:
    enabled: true
    model_name: "vinai/phobert-base-v2"
    top_k: 80
    weight: 0.7
  
  cross_encoder:
    enabled: true
    model_name: "vinai/phobert-large"
    top_k: 20
    weight: 0.3

app:
  top_k_retrieval: 100
  top_k_final: 10
  cache_questions_ttl_seconds: 3600
  cache_pipeline_ttl_seconds: 7200
```

**Pydantic schemas:**
```python
# config/schemas.py
class PathsConfig(BaseModel):
    root_dir: Path
    data_dir: Path
    model_dir: Path
    reports_dir: Path

class ContrastiveLearningConfig(BaseModel):
    enabled: bool
    loss_type: str
    max_length: PositiveInt
    batch_size: PositiveInt
    epochs: PositiveInt

class BiEncoderConfig(BaseModel):
    model_name: str
    enhancement_techniques: Dict[str, Any]
    training_params: Dict[str, Any]

class Config(BaseModel):
    paths: PathsConfig
    bi_encoder: BiEncoderConfig
    reranker_pipeline: RerankerPipelineConfig
    app: AppConfig
    logging: LoggingConfig
```

#### 6.3.2 Configuration Loader với Path Resolution

**Mục đích:** Load và resolve configuration với absolute paths

**Code logic (pseudocode):**
```python
def load_config(config_path: Optional[Path] = None) -> Config:
    global _config
    if _config:
        return _config
    
    # Determine project root và default config path
    project_root = Path(__file__).parent.parent
    if config_path is None:
        config_path = project_root / "config" / "default.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    # Load YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    # Resolve paths to be absolute
    config_data = _resolve_paths(config_data, project_root)
    
    # Validate với Pydantic
    try:
        _config = Config(**config_data)
        return _config
    except Exception as e:
        print(f"Configuration validation error: {e}")
        raise

def _resolve_paths(config_data: dict, root_dir: Path) -> dict:
    """Resolve all relative paths trong config thành absolute paths"""
    paths_config = config_data.get("paths", {})
    for key, value in paths_config.items():
        if isinstance(value, str):
            # Prepend root_dir, then resolve để handle '..' etc.
            paths_config[key] = str((root_dir / value).resolve())
    
    # Add root directory vào paths config
    paths_config["root_dir"] = str(root_dir)
    
    return config_data
```

### 6.4 Error Handling và Logging

#### 6.4.1 Comprehensive Error Handling Strategy

**Mục đích:** Xử lý lỗi một cách graceful với proper logging

**Error handling strategy:**
```python
# Error classification
class LawBotError(Exception):
    """Base exception cho LawBot"""
    pass

class PipelineError(LawBotError):
    """Pipeline-related errors"""
    pass

class ModelError(LawBotError):
    """Model loading/execution errors"""
    pass

class ConfigurationError(LawBotError):
    """Configuration errors"""
    pass

# Error handling decorator
def handle_pipeline_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PipelineError as e:
            logger.error(f"Pipeline error in {func.__name__}: {e}")
            # Return fallback results
            return []
        except ModelError as e:
            logger.error(f"Model error in {func.__name__}: {e}")
            # Try to reload model
            return handle_model_error(e)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            # Return empty results
            return []
    return wrapper

# Usage example
@handle_pipeline_errors
def predict(self, query, top_k_final=10):
    # Pipeline logic here
    pass
```

#### 6.4.2 Structured Logging với Logging Manager

**Mục đích:** Quản lý logging một cách có cấu trúc và configurable

**Logging manager:**
```python
# core/utils/logging_manager.py
def setup_logging(log_type: str = "app", workflow_timestamp: Optional[str] = None):
    """Setup logging với proper configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if workflow_timestamp:
        log_filename = f"{log_type}_{workflow_timestamp}.log"
    else:
        log_filename = f"{log_type}_{timestamp}.log"
    
    log_filepath = log_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filepath

def get_logger(name: str) -> logging.Logger:
    """Get logger với proper naming convention"""
    return logging.getLogger(f"LawBot.{name}")

# Usage example
logger = get_logger("pipeline")
logger.info("Pipeline initialized successfully")
logger.error("Pipeline failed", exc_info=True)
```

### 6.5 Caching và Performance Optimization

#### 6.5.1 Streamlit Caching Strategy

**Mục đích:** Tối ưu hóa performance với intelligent caching

**Caching levels:**
```python
# 1. Data caching (1 hour TTL)
@st.cache_data(ttl=3600)
def load_random_questions() -> Tuple[List[str], List[str]]:
    """Load random questions từ datasets"""
    # Implementation here
    pass

# 2. Resource caching (2 hours TTL)
@st.cache_resource(ttl=7200)
def load_pipeline(force_cpu=False):
    """Load và cache pipeline để tránh reloading"""
    # Implementation here
    pass

# 3. Computation caching (30 minutes TTL)
@st.cache_data(ttl=1800)
def run_comprehensive_evaluation(_pipeline, test_queries=None):
    """Run comprehensive evaluation với caching"""
    # Implementation here
    pass

# 4. File-based caching
def load_latest_comprehensive_evaluation():
    """Load evaluation results từ file cache"""
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return None
        
        # Find latest file
        comp_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
        if not comp_files:
            return None
        
        latest_file = max(comp_files, key=lambda p: p.stat().st_mtime)
        
        # Check if file is recent (within last 24 hours)
        file_age = time.time() - latest_file.stat().st_mtime
        if file_age > 86400:  # 24 hours
            return None
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data.get("results")
        
    except Exception as e:
        logger.error(f"Failed to load evaluation results: {e}")
        return None
```

#### 6.5.2 Memory Management và Optimization

**Mục đích:** Quản lý memory hiệu quả cho large models

**Memory optimization techniques:**
```python
# 1. Lazy loading
def get_pipeline_lazy():
    """Lazy load pipeline chỉ khi cần thiết"""
    try:
        # Validate parent law mapping
        if ensure_parent_law_mapping():
            logger.info("✅ Parent law mapping validated")
        
        pipeline = LegalQAPipeline()
        return pipeline
        
    except Exception as e:
        logger.warning(f"⚠️ Pipeline not available: {e}")
        return None

# 2. Memory cleanup
def cleanup_pipeline_resources(pipeline):
    """Cleanup resources để giải phóng memory"""
    try:
        if hasattr(pipeline, 'retriever'):
            pipeline.retriever.cleanup()
        if hasattr(pipeline, 'reranker'):
            pipeline.reranker.cleanup()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Pipeline cleanup completed")
        
    except Exception as e:
        logger.warning(f"Error during pipeline cleanup: {e}")

# 3. Batch processing
def process_queries_in_batches(queries, batch_size=5):
    """Process queries theo batches để tối ưu memory"""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        
        # Process batch
        batch_results = process_batch(batch_queries)
        results.extend(batch_results)
        
        # Memory cleanup after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
```

### 6.6 Data Processing và Transformation

#### 6.6.1 Unified Text Processing Pipeline

**Mục đích:** Xử lý text một cách nhất quán với legal domain expertise

**Text processing pipeline:**
```python
# core/transforms/base.py
class TextProcessor(BaseTransform):
    """Unified text processor cho cleaning, normalizing, và transforming text"""
    
    def __init__(self, normalize_unicode=True, lowercase=False, 
                 remove_punctuation=False, remove_extra_spaces=True):
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_spaces = remove_extra_spaces
    
    def __call__(self, text: str) -> str:
        """Apply configured text transformations"""
        if not isinstance(text, str):
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        
        # Lowercase conversion
        if self.lowercase:
            text = text.lower()
        
        # Punctuation removal
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        
        # Extra spaces removal
        if self.remove_extra_spaces:
            text = re.sub(r"\s+", " ", text).strip()
        
        return text

class LegalTextCleaner(TextProcessor):
    """Specialized text processor cho Vietnamese legal documents"""
    
    def __init__(self):
        super().__init__(
            normalize_unicode=True,
            lowercase=False,  # Legal text có meaningful capitalization
            remove_punctuation=False,  # Punctuation quan trọng trong legal text
            remove_extra_spaces=True
        )
        
        # Pre-compiled regex patterns
        self.article_pattern = re.compile(
            r"(Điều|Khoản|Điểm)\s+\d+[a-z]?\.?", re.IGNORECASE
        )
        
        # Legal abbreviations mapping
        self.abbreviation_map = {
            "QH": "Quốc hội",
            "UBTVQH": "Ủy ban Thường vụ Quốc hội",
            "CP": "Chính phủ",
            "NĐ-CP": "Nghị định - Chính phủ",
            "TTg": "Thủ tướng Chính phủ"
        }
    
    def __call__(self, text: str) -> str:
        """Apply base processing và legal-specific cleaning rules"""
        # Base processing first
        text = super().__call__(text)
        
        # Remove article/clause/point numbers
        text = self.article_pattern.sub(r"\1", text)
        
        # Replace common abbreviations
        for abbr, full_text in self.abbreviation_map.items():
            text = re.sub(rf"\b{abbr}\b", full_text, text)
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
```

#### 6.6.2 Dataset Management và Processing

**Mục đích:** Quản lý datasets một cách hiệu quả với proper validation

**Dataset classes:**
```python
# core/datasets/legal_qa.py
class BiEncoderDataset(Dataset):
    """Dataset cho training bi-encoder models"""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, 
                 max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"BiEncoderDataset initialized với {len(data)} examples, max_length={max_length}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        query = item.get("query", "")
        positive = item.get("positive", "")
        negative = item.get("negative", "")
        
        # Tokenize each part separately cho bi-encoder training
        query_inputs = self.tokenizer(
            query, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        positive_inputs = self.tokenizer(
            positive, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        negative_inputs = self.tokenizer(
            negative, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_inputs["input_ids"].squeeze(0),
            "query_attention_mask": query_inputs["attention_mask"].squeeze(0),
            "positive_input_ids": positive_inputs["input_ids"].squeeze(0),
            "positive_attention_mask": positive_inputs["attention_mask"].squeeze(0),
            "negative_input_ids": negative_inputs["input_ids"].squeeze(0),
            "negative_attention_mask": negative_inputs["attention_mask"].squeeze(0)
        }

class RerankerDataset(Dataset):
    """Dataset cho training cross-encoder reranker models"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, 
                 max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"RerankerDataset initialized với {len(data)} examples, max_length={max_length}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        query = item.get("query", "")
        passage = item.get("passage", "")
        label = item.get("label", 0)
        
        # Tokenize query-passage pair cho cross-encoder training
        inputs = self.tokenizer(
            query, passage, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }
```

---

### 6.7 Testing và Quality Assurance

#### 6.7.1 Unit Testing Strategy

**Mục đích:** Đảm bảo code quality với comprehensive testing

**Testing structure:**
```python
# tests/test_pipeline.py
import pytest
from unittest.mock import Mock, patch
from core.pipeline import LegalQAPipeline

class TestLegalQAPipeline:
    """Test cases cho LegalQAPipeline"""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline cho testing"""
        with patch('core.pipeline.RetrievalEngine') as mock_retriever, \
             patch('core.pipeline.RerankingEngine') as mock_reranker:
            
            mock_retriever.return_value.is_ready = True
            mock_reranker.return_value.is_ready = True
            
            pipeline = LegalQAPipeline()
            yield pipeline
    
    def test_pipeline_initialization(self, mock_pipeline):
        """Test pipeline initialization"""
        assert mock_pipeline.is_ready == True
        assert hasattr(mock_pipeline, 'retriever')
        assert hasattr(mock_pipeline, 'reranker')
    
    def test_predict_method(self, mock_pipeline):
        """Test predict method"""
        # Mock retriever response
        mock_pipeline.retriever.retrieve.return_value = [
            {'aid': 'test1', 'content': 'test content', 'retrieval_score': 0.8}
        ]
        
        # Mock reranker response
        mock_pipeline.reranker.rank_light.return_value = [
            {'aid': 'test1', 'content': 'test content', 'retrieval_score': 0.8, 
             'light_reranker_score': 0.9}
        ]
        
        results = mock_pipeline.predict("test query", top_k_final=1)
        
        assert len(results) == 1
        assert results[0]['aid'] == 'test1'
        assert 'final_score' in results[0]
    
    def test_pipeline_not_ready_error(self, mock_pipeline):
        """Test error khi pipeline chưa ready"""
        mock_pipeline.is_ready = False
        
        with pytest.raises(RuntimeError, match="Pipeline is not ready"):
            mock_pipeline.predict("test query")
```

#### 6.7.2 Integration Testing

**Mục đích:** Test integration giữa các components

**Integration test structure:**
```python
# tests/test_integration.py
import pytest
from core.pipeline import LegalQAPipeline
from core.retrieval import RetrievalEngine
from core.reranking import RerankingEngine

class TestPipelineIntegration:
    """Test integration giữa pipeline components"""
    
    @pytest.fixture
    def real_pipeline(self):
        """Create real pipeline với minimal models cho integration testing"""
        # Load pipeline với test configuration
        pipeline = LegalQAPipeline(
            bi_encoder_path="test_models/mini_bi_encoder",
            reranker_paths="test_models/mini_reranker"
        )
        return pipeline
    
    def test_end_to_end_pipeline(self, real_pipeline):
        """Test end-to-end pipeline execution"""
        # Test query processing
        query = "Quy định về xử phạt vi phạm giao thông"
        
        # Execute pipeline
        results = real_pipeline.predict(query, top_k_final=3)
        
        # Validate results structure
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert 'aid' in result
            assert 'content' in result
            assert 'final_score' in result
            assert 'retrieval_score' in result
    
    def test_tier_communication(self, real_pipeline):
        """Test communication giữa các tiers"""
        query = "Thủ tục đăng ký kinh doanh"
        
        # Test Tier 1 → Tier 2 communication
        tier1_results = real_pipeline.retriever.retrieve(query, top_k=10)
        assert len(tier1_results) == 10
        
        # Test Tier 2 → Tier 3 communication
        if real_pipeline.reranker:
            tier2_results = real_pipeline.reranker.rank_light(query, tier1_results[:5])
            assert len(tier2_results) == 5
            assert all('light_reranker_score' in doc for doc in tier2_results)
```

### 6.8 Code Quality và Best Practices

#### 6.8.1 Code Style và Documentation

**Mục đích:** Đảm bảo code quality với proper documentation và style

**Documentation standards:**
```python
# Example of well-documented code
class LegalQAPipeline:
    """
    Main pipeline orchestrator cho LawBot system.
    
    This class manages the 3-tier architecture:
    - Tier 1: Bi-Encoder Retrieval
    - Tier 2: Light Reranking  
    - Tier 3: Cross-Encoder Ensemble
    
    Attributes:
        retriever (RetrievalEngine): Tier 1 retrieval engine
        reranker (RerankingEngine): Tier 2 & 3 reranking engine
        is_ready (bool): Pipeline readiness status
        config (Config): Configuration object
    
    Example:
        >>> pipeline = LegalQAPipeline()
        >>> results = pipeline.predict("legal question", top_k_final=5)
        >>> print(f"Found {len(results)} results")
    """
    
    def __init__(self, bi_encoder_path: Optional[str] = None, 
                 reranker_paths: Optional[Union[str, List[str]]] = None):
        """
        Initialize LegalQAPipeline.
        
        Args:
            bi_encoder_path: Path to bi-encoder model
            reranker_paths: Path(s) to reranker model(s)
        
        Raises:
            RuntimeError: If pipeline initialization fails
            FileNotFoundError: If required model files not found
        """
        # Implementation here
        pass
```

#### 6.8.2 Performance Monitoring và Profiling

**Mục đích:** Monitor và optimize code performance

**Performance monitoring:**
```python
# core/utils/performance_monitor.py
import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def performance_monitor(func: Callable) -> Callable:
    """Decorator để monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = get_memory_usage()
            
            duration = end_time - start_time
            memory_diff = end_memory - start_memory
            
            logger.info(f"Function {func.__name__} took {duration:.3f}s, "
                       f"memory change: {memory_diff:.2f}MB")
    
    return wrapper

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Usage example
@performance_monitor
def predict(self, query: str, top_k_final: int = 10) -> List[Dict]:
    # Pipeline logic here
    pass
```

---

## 7. HƯỚNG DẪN VẬN HÀNH

### 7.1 Cài đặt và Setup

#### 7.1.1 Environment Setup

**Mục đích:** Hướng dẫn cài đặt môi trường cho LawBot

**Requirements:**
```bash
# Python version
Python 3.8+

# CUDA support (optional, for GPU acceleration)
CUDA 11.0+ (for PyTorch GPU support)

# System requirements
RAM: 16GB+ (32GB recommended)
Storage: 50GB+ free space
GPU: NVIDIA GPU với 8GB+ VRAM (optional)
```

**Installation steps:**
```bash
# 1. Clone repository
git clone https://github.com/your-org/LawBot.git
cd LawBot

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install additional dependencies for development
pip install -r requirements-dev.txt
```

#### 7.1.2 Configuration Setup

**Mục đích:** Cấu hình hệ thống cho môi trường cụ thể

**Configuration files:**
```yaml
# config/local.yml (override default.yml)
paths:
  data_dir: "/path/to/your/data"
  model_dir: "/path/to/your/models"
  reports_dir: "/path/to/your/reports"

bi_encoder:
  model_name: "your-custom-bi-encoder"
  enhancement_techniques:
    contrastive_learning:
      enabled: true
      batch_size: 8  # Adjust based on your GPU memory

app:
  cache_pipeline_ttl_seconds: 3600
  force_cpu: false  # Set to true if no GPU available
```

**Environment variables:**
```bash
# Set environment variables
export LAWBOOT_CONFIG_PATH="/path/to/your/config.yml"
export LAWBOOT_DATA_PATH="/path/to/your/data"
export LAWBOOT_MODEL_PATH="/path/to/your/models"
export CUDA_VISIBLE_DEVICES="0"  # Use specific GPU
```

### 7.2 Workflow Execution

#### 7.2.1 Automated Workflow với Centralized Paths

**Mục đích:** Chạy toàn bộ training workflow một cách tự động với centralized path management

**Command execution:**
```bash
# Run complete workflow với preset
python run_workflow.py --preset full

# Available presets:
# - data_preparation: Chỉ chuẩn bị dữ liệu
# - bi_encoder: Chỉ train bi-encoder
# - light_ranking: Chỉ train light reranker
# - cross_encoder: Chỉ train cross-encoder
# - faiss_index: Chỉ tạo FAISS index
# - evaluation: Chỉ chạy evaluation
# - full: Toàn bộ workflow (recommended)

# Custom workflow với specific steps
python run_workflow.py --stages data_preparation bi_encoder light_ranking

# Run với force restart (bỏ qua checkpoints)
python run_workflow.py --preset full --force-restart

# List available stages và presets
python run_workflow.py --list-stages
```

**Centralized Path Management Features:**
```bash
# Automated data freshness validation
python run_workflow.py --preset full  # Tự động kiểm tra và re-run data_preparation nếu cần

# Check centralized path status
python -c "from config.paths import validate_training_data_paths; print(validate_training_data_paths())"

# Get training data paths
python -c "from config.paths import get_training_data_path; print(get_training_data_path('tier_1', 'bi_encoder'))"
```

**Workflow monitoring:**
```bash
# Monitor workflow progress
tail -f logs/workflow_20241201_143022.log

# Check workflow status
python run_workflow.py --list-stages

# Resume interrupted workflow (tự động qua checkpoints)
python run_workflow.py --preset full

# Force restart toàn bộ workflow
python run_workflow.py --preset full --force-restart
```

#### 7.2.2 Manual Workflow Execution

**Mục đích:** Chạy từng bước workflow một cách thủ công

**Step-by-step execution:**
```bash
# 1. Data preparation
python data_processing/run_preparation.py \
  --input_dir data/raw \
  --output_dir features/processed \
  --config config/default.yml

# 2. Bi-encoder training
python training/run_bi_encoder.py \
  --data_path features/processed \
  --output_dir models/bi_encoder \
  --config config/default.yml \
  --hpo_trials 15

# 3. Create FAISS index
python training/run_create_faiss_index.py \
  --model_path models/bi_encoder \
  --corpus_path features/processed \
  --output_dir models/faiss_index

# 4. Light reranker training
python training/run_light_ranking.py \
  --data_path features/processed \
  --output_dir models/light_reranker \
  --config config/default.yml

# 5. Cross-encoder training
python training/run_reranker.py \
  --data_path features/processed \
  --output_dir models/cross_encoder \
  --config config/default.yml

# 6. Evaluation
python evaluation/run_evaluation.py \
  --pipeline_config config/default.yml \
  --output_dir reports/evaluation
```

### 7.3 Application Deployment

#### 7.3.1 Streamlit App Launch

**Mục đích:** Khởi chạy ứng dụng web Streamlit

**Launch commands:**
```bash
# Basic launch
python run_app.py

# Launch với specific configuration
python run_app.py --config config/production.yml

# Launch với custom port
python run_app.py --port 8502

# Launch với specific host
python run_app.py --host 0.0.0.0 --port 8501

# Launch với debug mode
python run_app.py --debug

# Launch với CPU-only mode
python run_app.py --force_cpu
```

**Environment configuration:**
```bash
# Set Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch app
streamlit run app/app.py
```

#### 7.3.2 Production Deployment

**Mục đích:** Deploy ứng dụng trong môi trường production

**Docker deployment:**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health

# Launch application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  lawbot:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
    environment:
      - LAWBOOT_CONFIG_PATH=/app/config/production.yml
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 7.4 Monitoring và Maintenance

#### 7.4.1 System Health Monitoring

**Mục đích:** Monitor trạng thái hệ thống và performance

**Health check commands:**
```bash
# Check system status
python -c "
from core.utils.system_check import check_system_health
status = check_system_health()
print(f'System Health: {status}')
"

# Check model status
python -c "
from core.utils.system_check import check_model_status
models = check_model_status()
for name, status in models.items():
    print(f'{name}: {status}')
"

# Check FAISS index status
python -c "
from core.utils.system_check import check_faiss_status
faiss_status = check_faiss_status()
print(f'FAISS Status: {faiss_status}')
"
```

**Log monitoring:**
```bash
# Monitor application logs
tail -f logs/app_*.log

# Monitor training logs
tail -f logs/workflow_*.log

# Monitor error logs
grep -i "error\|exception" logs/*.log

# Monitor performance logs
grep -i "performance\|timing" logs/*.log
```

#### 7.4.2 Performance Optimization

**Mục đích:** Tối ưu hóa performance của hệ thống

**Performance tuning:**
```bash
# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Batch size optimization
python training/run_bi_encoder.py \
  --batch_size 32 \
  --gradient_accumulation_steps 4

# Memory cleanup
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA cache cleared')
"
```

**Resource monitoring:**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor memory usage
watch -n 1 'free -h'

# Monitor disk usage
df -h

# Monitor process resources
top -p $(pgrep -f "python.*run_app.py")
```

### 7.5 Troubleshooting

#### 7.5.1 Common Issues và Solutions

**Mục đích:** Giải quyết các vấn đề thường gặp

**Common issues:**
```bash
# Issue 1: CUDA out of memory
# Solution: Reduce batch size hoặc enable gradient accumulation
python training/run_bi_encoder.py --batch_size 8 --gradient_accumulation_steps 8

# Issue 2: Model loading failed
# Solution: Check model paths và permissions
ls -la models/
chmod -R 755 models/

# Issue 3: FAISS index not found
# Solution: Recreate FAISS index
python training/run_create_faiss_index.py \
  --model_path models/bi_encoder \
  --corpus_path features/processed

# Issue 4: Pipeline not ready
# Solution: Check model status và reload pipeline
python -c "
from core.pipeline import LegalQAPipeline
pipeline = LegalQAPipeline()
print(f'Pipeline ready: {pipeline.is_ready}')
"

# Issue 5: Workflow import errors (ĐÃ FIX)
# Solution: Sử dụng lệnh đã được fix
python run_workflow.py --preset full --force-restart

# Issue 6: Checkpoint conflicts (ĐÃ FIX)
# Solution: Sử dụng --force-restart để bỏ qua checkpoints cũ
python run_workflow.py --preset full --force-restart
```

**Debug mode:**
```bash
# Enable debug logging
export LAWBOOT_LOG_LEVEL=DEBUG

# Run với verbose output
python run_app.py --verbose

# Check configuration
python -c "
from config.loader import load_config
config = load_config()
print('Configuration loaded successfully')
print(f'Data directory: {config.paths.data_dir}')
"
```

#### 7.5.2 Recovery Procedures

**Mục đích:** Khôi phục hệ thống khi gặp sự cố

**Recovery steps:**
```bash
# 1. Check system status
python -c "from core.utils.system_check import check_system_health; print(check_system_health())"

# 2. Restart pipeline
python -c "
from core.pipeline import LegalQAPipeline
pipeline = LegalQAPipeline()
print('Pipeline restarted')
"

# 3. Clear cache
python -c "
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()
print('Cache cleared')
"

# 4. Restart application
pkill -f "streamlit.*run_app.py"
python run_app.py
```

---

## 8. ĐÁNH GIÁ VÀ KHUYẾN NGHỊ

### 8.1 Đánh giá tổng quan

#### 8.1.1 Điểm mạnh của hệ thống

**Kiến trúc và thiết kế:**
- **Kiến trúc 3 tầng hiện đại**: Mỗi tầng được tối ưu hóa cho nhiệm vụ cụ thể
- **Modular design**: Code được tổ chức theo modules rõ ràng, dễ maintain
- **Scalable architecture**: Có thể mở rộng thêm tiers hoặc models mới
- **Configuration-driven**: Dễ dàng thay đổi cấu hình mà không cần sửa code
- **Workflow automation**: Hệ thống workflow tự động với checkpoint management

**Centralized Path Management:**
- **Centralized configuration**: Tất cả đường dẫn được quản lý tập trung tại `config/paths.py`
- **Automated path discovery**: Tự động tìm thư mục processed data mới nhất
- **Data freshness validation**: Tự động kiểm tra tính mới của dữ liệu
- **Consistent path resolution**: Đảm bảo tất cả tiers sử dụng cùng nguồn dữ liệu
- **Fallback mechanisms**: Graceful fallback khi không tìm thấy dữ liệu mới

**MLOps và Engineering:**
- **Comprehensive logging**: Logging có cấu trúc và dễ debug
- **Error handling**: Xử lý lỗi graceful với recovery mechanisms
- **Performance monitoring**: Theo dõi hiệu suất real-time
- **Caching strategy**: Intelligent caching để tối ưu performance
- **Version control**: Quản lý phiên bản models và experiments

**Kỹ thuật Machine Learning:**
- **Contrastive Learning**: Sử dụng TripletLoss hiệu quả cho bi-encoder
- **Hard Negative Mining**: Tự động cải thiện chất lượng training data
- **ADAPT technique**: Domain adaptation cho pháp luật Việt Nam
- **Ensemble learning**: Kết hợp multiple models để tăng accuracy
- **Hyperparameter Optimization**: Sử dụng Optuna để tối ưu hóa

**Workflow và Automation:**
- **Automated pipeline**: Workflow tự động với stage dependencies
- **Checkpoint management**: Resume interrupted workflows
- **Error recovery**: Tự động xử lý lỗi và retry
- **Progress tracking**: Real-time progress monitoring
- **Consistent execution**: Đảm bảo workflow chạy giống manual execution
- **Data freshness automation**: Tự động re-run data_preparation khi cần thiết

#### 8.1.2 Điểm cần cải thiện

**Performance và Scalability:**
- **Memory usage**: Cần tối ưu hóa memory cho large models
- **Processing speed**: Có thể cải thiện tốc độ xử lý với parallel processing
- **Batch processing**: Cần implement batch processing cho multiple queries
- **Async processing**: Có thể sử dụng async/await để tăng throughput

**Data Management:**
- **Data versioning**: Cần implement data versioning system
- **Incremental updates**: Cần hỗ trợ cập nhật dữ liệu incrementally
- **Data validation**: Cần tăng cường validation cho input data
- **Data lineage**: Cần tracking data lineage từ raw data đến final results

**User Experience:**
- **Real-time feedback**: Cần cải thiện real-time feedback cho users
- **Personalization**: Có thể thêm personalization features
- **Multi-language support**: Cần hỗ trợ đa ngôn ngữ tốt hơn
- **Mobile optimization**: Cần tối ưu hóa cho mobile devices

### 8.2 Khuyến nghị cải thiện

#### 8.2.1 Kỹ thuật Machine Learning

**Model Architecture:**
```python
# 1. Implement attention mechanisms
class AttentionEnhancedBiEncoder(nn.Module):
    def __init__(self, base_model, attention_dim=768):
        super().__init__()
        self.base_model = base_model
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=8)
        
    def forward(self, input_ids, attention_mask):
        # Base model encoding
        base_output = self.base_model(input_ids, attention_mask)
        
        # Apply attention mechanism
        attended_output, _ = self.attention(
            base_output.last_hidden_state,
            base_output.last_hidden_state,
            base_output.last_hidden_state
        )
        
        return attended_output

# 2. Implement contrastive learning với multiple negatives
class MultiNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=16):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        
    def forward(self, query_emb, positive_emb, negative_embs):
        # Calculate similarities
        pos_sim = torch.cosine_similarity(query_emb, positive_emb, dim=1)
        neg_sims = torch.cosine_similarity(
            query_emb.unsqueeze(1), 
            negative_embs, 
            dim=2
        )
        
        # Multi-negative contrastive loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        
        return F.cross_entropy(logits / self.temperature, labels)
```

**Training Optimization:**
```python
# 1. Implement curriculum learning
class CurriculumLearningScheduler:
    def __init__(self, difficulty_levels=[0.3, 0.6, 1.0]):
        self.difficulty_levels = difficulty_levels
        self.current_level = 0
        
    def get_training_data(self, full_dataset):
        current_difficulty = self.difficulty_levels[self.current_level]
        
        # Filter data based on difficulty
        filtered_data = [
            item for item in full_dataset 
            if item['difficulty_score'] <= current_difficulty
        ]
        
        return filtered_data
    
    def advance_level(self):
        if self.current_level < len(self.difficulty_levels) - 1:
            self.current_level += 1
            return True
        return False

# 2. Implement advanced HPO với early stopping
class AdvancedHPO:
    def __init__(self, n_trials=50, patience=10):
        self.n_trials = n_trials
        self.patience = patience
        
    def objective(self, trial):
        params = {
            'learning_rate': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 3, 10),
            'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.1),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5)
        }
        
        # Train model với early stopping
        model = self._train_with_early_stopping(params)
        
        # Evaluate model
        val_score = self._evaluate_model(model)
        
        return val_score
```

#### 8.2.2 System Architecture

**Microservices Architecture:**
```python
# 1. Implement API gateway
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LawBot API", version="v8.3")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_cache: bool = True

@app.post("/api/v1/query")
async def process_query(request: QueryRequest):
    try:
        # Process query through pipeline
        results = await pipeline.predict_async(
            request.query, 
            top_k_final=request.top_k
        )
        
        return {
            "status": "success",
            "results": results,
            "query": request.query,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Implement async pipeline processing
class AsyncLegalQAPipeline:
    async def predict_async(self, query: str, top_k_final: int = 10):
        # Async Tier 1 processing
        tier1_task = asyncio.create_task(
            self._async_retrieval(query, top_k=100)
        )
        
        # Async Tier 2 processing (when Tier 1 completes)
        tier1_results = await tier1_task
        tier2_task = asyncio.create_task(
            self._async_light_reranking(query, tier1_results[:50])
        )
        
        # Async Tier 3 processing (when Tier 2 completes)
        tier2_results = await tier2_task
        tier3_results = await self._async_cross_encoding(
            query, tier2_results[:20]
        )
        
        # Combine results
        final_results = self._combine_scores(tier3_results)
        return final_results[:top_k_final]
```

**Caching và Performance:**
```python
# 1. Implement Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def redis_cache(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

# 2. Implement connection pooling
from contextlib import asynccontextmanager
import aiohttp

class ConnectionPool:
    def __init__(self, max_connections=100):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=self.max_connections)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    @asynccontextmanager
    async def acquire(self):
        async with self.semaphore:
            session = await self.get_session()
            yield session
```

#### 8.2.3 Data Management

**Data Versioning và Lineage:**
```python
# 1. Implement data versioning
import hashlib
from datetime import datetime

class DataVersioning:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.versions_file = data_dir / "versions.json"
        
    def create_version(self, data_path: Path, metadata: dict) -> str:
        # Calculate data hash
        data_hash = self._calculate_hash(data_path)
        
        # Create version info
        version_info = {
            "version_id": data_hash[:8],
            "timestamp": datetime.now().isoformat(),
            "data_path": str(data_path),
            "metadata": metadata,
            "hash": data_hash
        }
        
        # Save version info
        self._save_version(version_info)
        
        return version_info["version_id"]
    
    def _calculate_hash(self, file_path: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

# 2. Implement data lineage tracking
class DataLineageTracker:
    def __init__(self):
        self.lineage_graph = {}
        
    def track_transformation(self, input_data: str, output_data: str, 
                           transformation: str, metadata: dict):
        """Track data transformation lineage"""
        if output_data not in self.lineage_graph:
            self.lineage_graph[output_data] = {
                "inputs": [],
                "transformations": [],
                "metadata": {}
            }
        
        self.lineage_graph[output_data]["inputs"].append(input_data)
        self.lineage_graph[output_data]["transformations"].append({
            "type": transformation,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        })
    
    def get_lineage(self, data_id: str) -> dict:
        """Get complete lineage for data item"""
        return self.lineage_graph.get(data_id, {})
```

### 8.3 Roadmap phát triển

#### 8.3.1 Short-term improvements (3-6 months)

**Performance Optimization:**
- Implement async processing cho pipeline
- Optimize memory usage với gradient checkpointing
- Add batch processing cho multiple queries
- Implement advanced caching strategies

**User Experience:**
- Add real-time progress indicators
- Implement query suggestions và autocomplete
- Add result filtering và sorting options
- Improve mobile responsiveness

**Monitoring và Debugging:**
- Add comprehensive metrics dashboard
- Implement distributed tracing
- Add automated alerting system
- Improve error reporting và recovery

#### 8.3.2 Medium-term improvements (6-12 months)

**Advanced ML Techniques:**
- Implement transformer-based architectures
- Add multi-task learning capabilities
- Implement active learning strategies
- Add federated learning support

**System Architecture:**
- Migrate to microservices architecture
- Implement API gateway và load balancing
- Add horizontal scaling capabilities
- Implement service mesh

**Data Management:**
- Add data versioning system
- Implement data lineage tracking
- Add incremental data updates
- Implement data quality monitoring

#### 8.3.3 Long-term vision (1-2 years)

**AI Capabilities:**
- Implement multi-modal understanding (text + images)
- Add reasoning capabilities
- Implement conversational AI
- Add personalized recommendations

**Enterprise Features:**
- Multi-tenant architecture
- Advanced security features
- Compliance monitoring
- Integration với enterprise systems

**Research và Innovation:**
- Novel architectures cho legal AI
- Advanced NLP techniques
- Cross-lingual capabilities
- Explainable AI features

### 8.4 Kết luận

LawBot v8.3 đã đạt được những thành tựu đáng kể trong việc xây dựng hệ thống AI hỏi đáp pháp luật Việt Nam với kiến trúc 3 tầng hiện đại. Hệ thống đã tích hợp thành công các kỹ thuật machine learning tiên tiến như contrastive learning, hard negative mining, ADAPT domain adaptation, và ensemble learning.

**Những điểm nổi bật:**
- **Kiến trúc robust**: 3-tier architecture với mỗi tầng được tối ưu hóa
- **Kỹ thuật ML hiện đại**: Sử dụng các techniques mới nhất trong NLP
- **MLOps practices**: Comprehensive logging, monitoring, và error handling
- **User experience**: Giao diện Streamlit intuitive và responsive
- **Performance**: Tối ưu hóa cho speed và accuracy
- **Workflow automation**: Hệ thống workflow tự động với checkpoint management

**Hướng phát triển:**
- Tiếp tục cải thiện performance và scalability
- Thêm advanced ML techniques và architectures
- Mở rộng sang microservices architecture
- Tăng cường monitoring và observability
- Phát triển enterprise features

LawBot đã tạo nền tảng vững chắc cho việc phát triển hệ thống AI pháp luật trong tương lai, với khả năng mở rộng và thích ứng với các yêu cầu mới. Hệ thống workflow automation giờ đây hoạt động ổn định và đáng tin cậy.

---

## PHỤ LỤC

### A. Cấu hình chi tiết

### B. API Reference

### C. Troubleshooting Guide

### D. Performance Benchmarks

### E. Changelog

---

**Tài liệu này được tạo bởi LawBot Development Team**
**Phiên bản: v8.3 | Ngày cập nhật: 2024**
**Liên hệ: dev-team@lawbot.com**