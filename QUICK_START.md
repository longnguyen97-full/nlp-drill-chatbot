# LawBot v8.3 - Quick Start Guide (Comprehensive Optimization & Centralized Configuration)

## 🚀 **Cài đặt nhanh**

### **1. Clone & Setup**
```bash
# Clone repository
git clone https://github.com/your-org/lawbot.git
cd lawbot

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### **2. Kiểm tra environment**
```bash
# Kiểm tra Python version
python --version  # Python 3.8+

# Kiểm tra CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Kiểm tra dependencies
python -c "import sentence_transformers, faiss, optuna; print('All packages installed')"
```

## 🔄 **Workflow Commands**

### **1. Automated Pipeline (Recommended)**
```bash
# Chạy toàn bộ pipeline với Contrastive Learning
python run_workflow.py --preset full

# Chạy training stages only (Bi-Encoder + Light Ranking + Cross-Encoder)
python run_workflow.py --preset training

# Chạy retrieval stages only (Bi-Encoder + FAISS)
python run_workflow.py --preset retrieval

# Chạy evaluation only
python run_workflow.py --preset evaluation_only

# Force restart (ignore checkpoints)
python run_workflow.py --preset full --force-restart
```

### Lưu ý quan trọng về dữ liệu (v8.3)
- **Real-data only**: Hệ thống chỉ sử dụng dữ liệu thật được tạo bởi `data_processing/run_preparation.py`
- **No synthetic fallback**: Không còn cơ chế fallback sang synthetic; nếu thiếu dữ liệu thật, các stage training sẽ dừng với lỗi rõ ràng
- **Centralized Path Management**: Tất cả đường dẫn được quản lý tập trung tại `config/paths.py`
- **Centralized Model Configuration**: Tất cả cấu hình model được quản lý tập trung tại `config/models.py`
- **Automated Data Discovery**: Hệ thống tự động tìm thư mục processed data mới nhất với timestamp (ví dụ: `features/processed_data_20250820_134752`)
- **Data Freshness Validation**: Workflow tự động kiểm tra tính mới của dữ liệu và re-run `data_preparation` khi cần thiết
- **Advanced HPO**: Hyperparameter optimization với Optuna, Bayesian search, và early stopping
- **Comprehensive Evaluation**: Multi-tier evaluation với precision, recall, F1, NDCG, MRR, quality metrics
- **Unified Reports Storage**: Consolidated evaluation reports trong single `reports/` directory

**Các file dữ liệu cần thiết:**
- `bi_encoder_train.jsonl` (Tier 1)
- `cross_encoder_train.jsonl` (Tier 3)
- `processed_corpus.json` (Tier 1 & 3)
- `training_data.jsonl` (Tier 2)
- `negative_pool.jsonl` (Tier 2)

Hệ thống sẽ tự động kiểm tra bằng `config/paths.validate_training_data_paths()` và log chi tiết trong `run_workflow.py`.

### **2. Manual Step-by-Step**
```bash
# Stage 1: Data Preparation
python data_processing/run_preparation.py

# Stage 2: Bi-Encoder Training
python training/run_bi_encoder.py

# Stage 3: Light Ranking Training
python training/run_light_ranking.py

# Stage 4: Cross-Encoder Training
python training/run_reranker.py

# Stage 5: FAISS Index Creation
python training/run_create_faiss_index.py

# Stage 6: Evaluation
python evaluation/run_evaluation.py
```

### **3. Custom Stage Selection**
```bash
# Chạy specific stages
python run_workflow.py --stages data_preparation bi_encoder

# Chạy với dependencies
python run_workflow.py --stages bi_encoder light_ranking cross_encoder

# Chạy evaluation với trained models
python run_workflow.py --stages evaluation
```

## 🧹 **Environment Management**

### **1. Cleanup & Reset**
```bash
# Xóa checkpoints
rm -rf checkpoints/*

# Xóa models cũ
rm -rf models/*

# Xóa features cũ
rm -rf features/*

# Xóa reports cũ
rm -rf reports/*

# Reset hoàn toàn
python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ['checkpoints', 'models', 'features', 'reports']]"
```

### **2. Selective Cleanup**
```bash
# Xóa specific model
rm -rf models/light-ranking_*

# Xóa specific checkpoint
rm -f checkpoints/light_ranking_training.json

# Xóa specific features
rm -f features/faiss_index.bin
```

### **3. Backup & Restore**
```bash
# Backup models
cp -r models models_backup_$(date +%Y%m%d_%H%M%S)

# Backup features
cp -r features features_backup_$(date +%Y%m%d_%H%M%S)

# Restore from backup
cp -r models_backup_20250816_221146/* models/
```

## 📊 **Monitoring & Debugging**

### **1. Real-time Monitoring**
```bash
# Monitor training progress
tail -f logs/training/*.log

# Monitor specific stage
tail -f logs/training/light_ranking_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

### **2. Log Analysis**
```bash
# Tìm errors
grep -r "ERROR\|Exception\|Failed" logs/

# Tìm training progress
grep -r "Epoch.*Loss" logs/training/

# Tìm evaluation results
grep -r "Accuracy\|F1\|NDCG" logs/evaluation/

# Tìm performance metrics
grep -r "Score\|Similarity\|Recall" logs/
```

### **3. Performance Testing**
```bash
# Test retrieval engine
python -c "from core.retrieval import RetrievalEngine; r = RetrievalEngine('models/bi-encoder_20250816_221146', 'features/faiss_index.bin', 'features/aid_map.json', 'features/index_to_aid.json'); print('Index info:', r.get_index_info())"

# Test pipeline
python -c "from core.pipeline import LegalQAPipeline; p = LegalQAPipeline(); results = p.predict('Luật về đất đai?', top_k_final=3); print(f'Results: {len(results)}')"

# Test model loading
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('models/bi-encoder_20250816_221146'); print('Model loaded:', m is not None)"

# Test comprehensive evaluation
python -c "from app.pages.analysis import run_comprehensive_evaluation; results = run_comprehensive_evaluation(None); print('Evaluation completed:', results is not None)"

# Test reports loading
python -c "from app.pages.analysis import load_latest_comprehensive_evaluation; results = load_latest_comprehensive_evaluation(); print('Reports loaded:', results is not None)"
```

## 🏷️ **Model Management**

### **1. Model Status Check**
```bash
# Kiểm tra model status
python -c "from core.utils.system_check import get_model_status; print('Models:', get_model_status())"

# Kiểm tra FAISS index
python -c "from core.utils.system_check import get_faiss_index_status; print('FAISS:', get_faiss_index_status())"

# Kiểm tra device info
python -c "from core.utils.system_check import get_device_info; print('Device:', get_device_info())"

# Kiểm tra centralized paths
python -c "from config.paths import validate_training_data_paths; print('Path validation:', validate_training_data_paths())"

# Kiểm tra centralized model configuration
python -c "from config.models import MODEL_TYPES, MODEL_STATUS_KEYS; print('Model Types:', list(MODEL_TYPES.keys())); print('Status Keys:', MODEL_STATUS_KEYS)"
```

### **2. Model Versioning**
```bash
# List model versions
ls -la models/

# Check model metadata
cat models/bi-encoder_20250816_221146/training_metadata.json

# Check HPO results
cat models/light-ranking_20250816_221255/hpo_results.json

# Check model size
du -sh models/*/
```

### **3. Model Comparison**
```bash
# Compare model performance
python evaluation/run_evaluation.py --compare-models

# Compare specific models
python evaluation/run_evaluation.py --model1 bi-encoder_20250816_221146 --model2 bi-encoder_20250816_203414
```

## 🔧 **Configuration & Tuning**

### **1. Centralized Configuration Management**

#### **1.1 Model Configuration (`config/models.py`)**
```bash
# Kiểm tra model types
python -c "from config.models import MODEL_TYPES; print('Available models:', [m['display_name'] for m in MODEL_TYPES.values()])"

# Kiểm tra model directory mapping
python -c "from config.models import MODEL_DIRECTORY_MAPPING; print('Model directories:', MODEL_DIRECTORY_MAPPING)"

# Lấy display name cho model
python -c "from config.models import get_display_name; print('Bi-Encoder display:', get_display_name('bi_encoder'))"

# Lấy model config
python -c "from config.models import get_model_config; print('Bi-Encoder config:', get_model_config('bi_encoder'))"
```

#### **1.2 Path Configuration (`config/paths.py`)**
```bash
# Validate training data paths
python -c "from config.paths import validate_training_data_paths; result = validate_training_data_paths(); print('Validation result:', result['overall']['status'])"

# Get training data path cho specific tier
python -c "from config.paths import get_training_data_path; print('Bi-Encoder path:', get_training_data_path('bi_encoder'))"

# Get latest processed data directory
python -c "from config.paths import get_latest_processed_data_dir; print('Latest data dir:', get_latest_processed_data_dir())"
```

### **2. HPO (Hyperparameter Optimization)**
```bash
# Run HPO cho Light Ranking
python training/run_light_ranking.py --hpo --trials 100

# Run HPO cho Cross-Encoder
python training/run_reranker.py --hpo --trials 150

# Run HPO với custom study
python training/run_light_ranking.py --hpo --study-name custom_study --trials 200

# Advanced HPO với Bayesian search
python training/run_bi_encoder.py --hpo --trials 50 --sampler tpe --pruner median

# HPO với early stopping
python training/run_light_ranking.py --hpo --trials 100 --early-stopping --patience 10

# Check HPO results
python -c "import json; results = json.load(open('models/light_ranking_latest/hpo_results.json')); print('Best params:', results['best_params']); print('Best score:', results['best_score'])"
```

### **2. Configuration Updates**
```bash
# Update config
python -c "from config.loader import config; print('Current config:', config.model_dump())"

# Validate config
python -c "from config.schemas import validate_config; validate_config()"

# Update specific settings
python -c "import yaml; config = yaml.safe_load(open('config/default.yml')); config['training']['batch_size'] = 32; yaml.dump(config, open('config/default.yml', 'w'))"

# Kiểm tra centralized paths
python -c "from config.paths import TRAINING_DATA_PATHS; print('Training paths:', TRAINING_DATA_PATHS)"

# Kiểm tra centralized model configuration
python -c "from config.models import MODEL_TYPES, MODEL_STATUS_KEYS; print('Model types:', list(MODEL_TYPES.keys())); print('Status keys:', MODEL_STATUS_KEYS)"

# Validate training data paths
python -c "from config.paths import validate_training_data_paths; result = validate_training_data_paths(); print('Path validation:', result['overall']['status'])"

# Get latest processed data directory
python -c "from config.paths import get_latest_processed_data_dir; print('Latest data dir:', get_latest_processed_data_dir())"
```

### **3. Environment Variables**
```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set log level
export LAWBOT_LOG_LEVEL=DEBUG

# Set model cache
export TRANSFORMERS_CACHE=/path/to/cache

# Set FAISS thread count
export OMP_NUM_THREADS=8
```

## 🚀 **Application Usage**

### **1. Start Streamlit App**
```bash
# Start app
python run_app.py

# Start với custom port
python run_app.py --port 8502

# Start với custom config
python run_app.py --config custom_config.yml
```

### **2. API Usage**
```bash
# Health check
curl http://localhost:8501/health

# Query processing
curl -X POST http://localhost:8501/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Luật về đất đai quy định gì?", "top_k": 5}'

# Model status
curl http://localhost:8501/api/status
```

### **3. Batch Processing**
```bash
# Process multiple queries
python -c "
from core.pipeline import LegalQAPipeline
p = LegalQAPipeline()
queries = ['Luật về đất đai?', 'Quyền sử dụng đất?', 'Thuế đất đai?']
for q in queries:
    results = p.predict(q, top_k_final=3)
    print(f'Query: {q}, Results: {len(results)}')
"
```

## 🐛 **Troubleshooting**

### **1. Common Issues**
```bash
# CUDA out of memory
export CUDA_VISIBLE_DEVICES=""
python training/run_light_ranking.py

# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python training/run_bi_encoder.py

# Port conflicts
lsof -ti:8501 | xargs kill -9
python run_app.py
```

### **2. Debug Commands**
```bash
# Check Python path
python -c "import sys; print('Python path:', sys.path)"

# Check CUDA installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check file permissions
ls -la models/ features/ logs/

# Check disk space
df -h
```

### **3. Recovery Procedures**
```bash
# Reset failed training
rm -f checkpoints/*_training.json
python training/run_light_ranking.py

# Rebuild FAISS index
rm -f features/faiss_index.bin
python training/run_create_faiss_index.py

# Restore from backup
cp -r models/backup/* models/
```

## 📈 **Performance Optimization**

### **1. GPU Optimization**
```bash
# Set optimal CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Monitor GPU memory
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### **2. Memory Optimization**
```bash
# Set optimal batch sizes
export LAWBOT_BATCH_SIZE=16
export LAWBOT_GRADIENT_ACCUMULATION_STEPS=4

# Monitor memory usage
watch -n 1 'free -h && echo "---" && df -h'
```

### **3. Parallel Processing**
```bash
# Run multiple stages in parallel
python training/run_bi_encoder.py &
python training/run_light_ranking.py &
wait

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
python training/run_reranker.py --multi-gpu
```

## 🔍 **Data Analysis**

### **1. Corpus Analysis**
```bash
# Analyze legal corpus
python -c "
import json
data = json.load(open('data/raw/legal_corpus.json'))
print('Total articles:', len(data))
print('Sample content:', data[0]['content'][:100] if 'content' in data[0] else 'N/A')
"

# Check data distribution
python -c "
import json
from collections import Counter
data = json.load(open('data/raw/legal_corpus.json'))
categories = [d.get('category', 'unknown') for d in data]
print('Category distribution:', Counter(categories))
"
```

### **2. Feature Analysis**
```bash
# Analyze FAISS index
python -c "
import faiss
index = faiss.read_index('features/faiss_index.bin')
print('Index size:', index.ntotal)
print('Dimensions:', index.d)
print('Is trained:', index.is_trained)
"

# Check embeddings quality
python -c "
import numpy as np
embeddings = np.load('features/processed_data_20250816_221131/embeddings.npy')
print('Embeddings shape:', embeddings.shape)
print('Embeddings stats:', {'mean': embeddings.mean(), 'std': embeddings.std()}
"
```

## 📚 **Documentation & Help**

### **1. Help Commands**
```bash
# Show workflow help
python run_workflow.py --help

# Show training help
python training/run_light_ranking.py --help

# Show evaluation help
python evaluation/run_evaluation.py --help
```

### **2. List Available Options**
```bash
# List workflow stages
python run_workflow.py --list-stages

# List presets
python run_workflow.py --list-presets

# List model types
python -c "from core.utils.system_check import get_model_status; print('Available models:', list(get_model_status().keys()))"
```

### **3. Version Information**
```bash
# Check LawBot version
python -c "import lawbot; print('LawBot version:', lawbot.__version__)"

# Check all package versions
pip list | grep -E "(torch|transformers|faiss|optuna|streamlit)"

# Check git status
git status
git log --oneline -5
```

---

**🚀 Ready to use LawBot v8.2!** 

For detailed information, see [README.md](README.md).
For advanced usage, see the training and evaluation documentation.
