import streamlit as st
import sys
import os
import json
import random
import time
import gc
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

# Them thu muc goc vao path de import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from core.pipeline import LegalQAPipeline


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def load_random_questions() -> Tuple[List[str], List[str]]:
    """
    Load random questions from training and test datasets
    Returns: (training_questions, test_questions)
    """
    training_questions = []
    test_questions = []

    try:
        # Load training questions
        train_file = os.path.join(config.DATA_RAW_DIR, "train.json")
        if os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
                training_questions = [
                    item.get("question", "")
                    for item in train_data
                    if item.get("question")
                ]

        # Load test questions
        test_file = os.path.join(config.DATA_RAW_DIR, "public_test.json")
        if os.path.exists(test_file):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
                test_questions = [
                    item.get("question", "")
                    for item in test_data
                    if item.get("question")
                ]

    except Exception as e:
        st.warning(f"Không thể load câu hỏi từ datasets: {e}")

    return training_questions, test_questions


@st.cache_resource(ttl=1800)  # Cache for 30 minutes
def load_pipeline(force_cpu=False):
    """Tai va cache pipeline de tranh load lai moi lan tuong tac."""
    try:
        # Set CUDA environment variables to avoid issues
        import os

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Force CPU if requested
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            st.info("Đang sử dụng CPU mode")

        # Memory cleanup before loading
        gc.collect()

        start_time = time.time()
        pipeline = LegalQAPipeline()
        load_time = time.time() - start_time

        if not pipeline.is_ready:
            st.error(
                "Loi khoi tao Pipeline. Vui long kiem tra logs o terminal de biet chi tiet."
            )
            st.warning(
                "Hay chac chan rang ban da huan luyen cac mo hinh va dat chung vao thu muc 'models', sau do chay 'scripts/04_build_faiss_index.py'."
            )
            return None

        st.success(f"✅ Pipeline loaded successfully in {load_time:.2f}s")
        return pipeline

    except Exception as e:
        st.error(f"Lỗi khởi tạo pipeline: {e}")
        st.info("Thử khởi động lại app hoặc kiểm tra logs")
        return None


def calculate_optimal_parameters(
    final_results_count: int, search_aggressiveness: str = "balanced"
) -> Dict[str, int]:
    """
    Tính toán các tham số tối ưu cho tầng 1 và tầng 2 dựa trên số kết quả cuối cùng
    và mức độ aggressive của tìm kiếm
    """
    # Multiplier based on search aggressiveness
    multipliers = {
        "conservative": {"retrieval": 10, "light": 3},
        "balanced": {"retrieval": 20, "light": 4},
        "aggressive": {"retrieval": 30, "light": 6},
    }

    mult = multipliers.get(search_aggressiveness, multipliers["balanced"])

    # Tầng 1 (Retrieval): Lấy nhiều hơn để có đủ candidate cho tầng 2
    top_k_retrieval = max(50, final_results_count * mult["retrieval"])

    # Tầng 2 (Light Reranking): Lấy vừa đủ để tầng 3 có thể xử lý
    top_k_light_reranking = max(20, final_results_count * mult["light"])

    return {
        "top_k_retrieval": top_k_retrieval,
        "top_k_light_reranking": top_k_light_reranking,
        "top_k_final": final_results_count,
    }


def get_random_question(questions: List[str], question_type: str) -> str:
    """Get a random question from the specified list"""
    if not questions:
        return f"Không có câu hỏi {question_type} nào"
    return random.choice(questions)


def display_performance_metrics(start_time: float, end_time: float, results_count: int):
    """Display performance metrics"""
    duration = end_time - start_time

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏱️ Thời gian xử lý", f"{duration:.2f}s")
    with col2:
        st.metric("📊 Kết quả tìm thấy", results_count)
    with col3:
        if duration > 0:
            st.metric("⚡ Tốc độ", f"{results_count/duration:.1f} kết quả/s")


# Page configuration
st.set_page_config(
    page_title="Hệ thống Hỏi-Đáp Pháp luật",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add navigation
st.sidebar.title("🎯 Legal QA System")
page = st.sidebar.selectbox(
    "Chọn trang",
    ["🔍 Tìm kiếm", "📊 Kết quả Training", "ℹ️ Thông tin hệ thống"],
    help="Chọn trang để xem",
)

if page == "🔍 Tìm kiếm":
    # Main search functionality
    st.title("🔍 Legal QA - Hệ thống Hỏi đáp Pháp luật")
    st.markdown("Hệ thống AI hỗ trợ tìm kiếm và trả lời câu hỏi pháp luật Việt Nam")

    # Sidebar configuration
    with st.sidebar.expander("⚙️ Cấu hình tìm kiếm", expanded=True):
        st.markdown("### 🎯 Tùy chỉnh kết quả")
        final_results_count = st.slider(
            "Số lượng kết quả cuối cùng",
            min_value=1,
            max_value=50,
            value=3,
            help="Số lượng kết quả sẽ hiển thị (Tầng 3)",
        )
        search_aggressiveness = st.selectbox(
            "Mức độ tìm kiếm",
            options=["conservative", "balanced", "aggressive"],
            index=1,
            help="Conservative: Ít kết quả, chính xác cao | Balanced: Cân bằng | Aggressive: Nhiều kết quả, đa dạng",
        )
        force_cpu = st.checkbox(
            "Sử dụng CPU",
            value=False,
            help="Bắt buộc sử dụng CPU thay vì GPU (chậm hơn nhưng ổn định)",
        )

    # Random question feature
    training_questions, test_questions = load_random_questions()
    if training_questions or test_questions:
        with st.sidebar.expander("🎲 Câu hỏi ngẫu nhiên", expanded=True):
            question_source = st.selectbox(
                "Chọn nguồn câu hỏi",
                options=["Training Dataset", "Test Dataset", "Cả hai"],
                help="Câu hỏi từ training dataset thường đa dạng hơn",
            )
            if st.button("🎲 Lấy câu hỏi ngẫu nhiên"):
                # Reset the used flag when getting a new random question
                if hasattr(st.session_state, "random_question_used"):
                    del st.session_state.random_question_used
                if question_source == "Training Dataset" and training_questions:
                    random_q = get_random_question(training_questions, "training")
                    st.session_state.random_question = random_q
                    st.session_state.current_question = random_q
                elif question_source == "Test Dataset" and test_questions:
                    random_q = get_random_question(test_questions, "test")
                    st.session_state.random_question = random_q
                    st.session_state.current_question = random_q
                elif question_source == "Cả hai":
                    all_questions = training_questions + test_questions
                    random_q = get_random_question(all_questions, "combined")
                    st.session_state.random_question = random_q
                    st.session_state.current_question = random_q
                else:
                    st.warning("Không có câu hỏi nào trong dataset đã chọn")
    else:
        st.info("📝 Không tìm thấy datasets để load câu hỏi ngẫu nhiên")

    # Architecture info
    with st.expander("ℹ️ Kiến trúc hệ thống (3 tầng)"):
        st.markdown(
            """
        **🎯 Tầng 1 - Bi-Encoder Retrieval:** Tìm kiếm nhanh ứng viên ban đầu  
        **⚡ Tầng 2 - Light Reranker:** Lọc xuống ứng viên chất lượng cao  
        **🎯 Tầng 3 - Cross-Encoder Reranking:** Xếp hạng chính xác kết quả cuối cùng
        """
        )

    # Load pipeline
    pipeline = load_pipeline(force_cpu=force_cpu)

    if pipeline:
        # Initialize current question in session state if not exists
        if "current_question" not in st.session_state:
            st.session_state.current_question = (
                "Người lao động được nghỉ phép bao nhiêu ngày?"
            )

        # Update current question if random question is available
        if hasattr(st.session_state, "random_question") and not hasattr(
            st.session_state, "random_question_used"
        ):
            st.session_state.current_question = st.session_state.random_question
            st.session_state.random_question_used = True

        query = st.text_input(
            "Nhập câu hỏi của bạn:",
            value=st.session_state.current_question,
            help="Nhập câu hỏi về pháp luật hoặc sử dụng tính năng câu hỏi ngẫu nhiên",
            key="query_input",
        )

        # Update current question when user types something new
        if query != st.session_state.current_question:
            st.session_state.current_question = query
            # Clear random question flags when user manually changes the question
            if hasattr(st.session_state, "random_question"):
                del st.session_state.random_question
            if hasattr(st.session_state, "random_question_used"):
                del st.session_state.random_question_used

        # Initialize searching state
        if "searching" not in st.session_state:
            st.session_state.searching = False

        # Search button with disabled state while processing
        search_clicked = st.button(
            "🔍 Tìm kiếm", type="primary", disabled=st.session_state.searching
        )

        if search_clicked and not st.session_state.searching:
            if query:
                st.session_state.searching = True
                try:
                    # Calculate optimal parameters
                    params = calculate_optimal_parameters(
                        final_results_count, search_aggressiveness
                    )

                    # Display calculated parameters
                    with st.expander("📊 Thông số tìm kiếm được tính toán tự động"):
                        st.markdown(
                            f"**🎯 Tầng 1 - Retrieval:** {params['top_k_retrieval']} ứng viên"
                        )
                        st.markdown(
                            f"**⚡ Tầng 2 - Light Reranking:** {params['top_k_light_reranking']} ứng viên"
                        )
                        st.markdown(
                            f"**🎯 Tầng 3 - Final Reranking:** {params['top_k_final']} kết quả cuối cùng"
                        )
                        st.markdown(f"**⚙️ Mức độ tìm kiếm:** {search_aggressiveness}")

                    # Performance monitoring
                    start_time = time.time()
                    with st.spinner("🔎 Đang tìm kiếm... vui lòng đợi"):
                        # Get results from pipeline
                        results = pipeline.predict(
                            query,
                            top_k_retrieval=params["top_k_retrieval"],
                            top_k_final=params["top_k_final"],
                            top_k_light_reranking=params["top_k_light_reranking"],
                        )
                    end_time = time.time()

                    if results:
                        # Display performance metrics
                        display_performance_metrics(start_time, end_time, len(results))

                        # Display results in tabs
                        tab1, tab2 = st.tabs(["📄 Xem chi tiết", "📊 So sánh điểm số"])
                        with tab1:
                            st.success(f"✅ Tìm thấy {len(results)} kết quả phù hợp!")
                            for i, result in enumerate(results, 1):
                                with st.expander(
                                    f"#{i} - {result['aid']} (Điểm: {result['rerank_score']:.3f})",
                                    expanded=(i <= 3),
                                ):
                                    st.markdown(f"**ID:** {result['aid']}")
                                    st.markdown(f"**Điểm retrieval:** {result['retrieval_score']:.3f}")
                                    st.markdown(f"**Điểm rerank:** {result['rerank_score']:.3f}")
                                    st.markdown("**Nội dung:**")
                                    st.text(result["content"])
                        with tab2:
                            # Score comparison chart
                            scores_data = {
                                "Kết quả": [f"#{i+1}" for i in range(len(results))],
                                "Điểm Rerank": [r["rerank_score"] for r in results],
                                "Điểm Retrieval": [r["retrieval_score"] for r in results],
                            }
                            df_scores = pd.DataFrame(scores_data)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Điểm cao nhất", f"{max(scores_data['Điểm Rerank']):.3f}")
                            with col2:
                                st.metric("Điểm thấp nhất", f"{min(scores_data['Điểm Rerank']):.3f}")
                            with col3:
                                st.metric(
                                    "Điểm trung bình",
                                    f"{sum(scores_data['Điểm Rerank'])/len(scores_data['Điểm Rerank']):.3f}",
                                )
                            st.bar_chart(df_scores.set_index("Kết quả"))
                    else:
                        st.warning(
                            "❌ Không tìm thấy kết quả phù hợp. Hãy thử câu hỏi khác."
                        )
                except Exception as e:
                    st.error(f"❌ Lỗi khi tìm kiếm: {str(e)}")
                    st.info("💡 Gợi ý: Thử giảm số lượng kết quả hoặc sử dụng CPU mode")
                finally:
                    st.session_state.searching = False
    else:
        st.error("❌ Không thể tải pipeline. Vui lòng kiểm tra lại hệ thống.")

elif page == "📊 Kết quả Training":
    st.title("📊 Kết quả Training Chi tiết")
    st.markdown("Tổng hợp kết quả training và đánh giá hiệu suất từng tầng")

    # Load training results
    @st.cache_data(ttl=1800)
    def load_training_results():
        """Load training results from reports"""
        results = {}

        # Load latest evaluation report
        reports_dir = Path("reports")
        if reports_dir.exists():
            report_files = list(reports_dir.glob("evaluation_report_*.json"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_report, "r", encoding="utf-8") as f:
                        results["evaluation"] = json.load(f)
                except Exception as e:
                    st.error(f"Không thể load evaluation report: {e}")

        # Load model info
        models_dir = Path("models")
        if models_dir.exists():
            results["models"] = {}
            for model_type in ["bi-encoder", "cross-encoder", "light-reranker"]:
                model_path = models_dir / model_type
                if model_path.exists():
                    results["models"][model_type] = {
                        "exists": True,
                        "size_mb": sum(
                            f.stat().st_size
                            for f in model_path.rglob("*")
                            if f.is_file()
                        )
                        / (1024 * 1024),
                    }
                else:
                    results["models"][model_type] = {"exists": False}

        return results

    results = load_training_results()

    # Display overview
    col1, col2, col3 = st.columns(3)

    with col1:
        if results.get("models", {}).get("bi-encoder", {}).get("exists"):
            st.success("✅ Bi-Encoder")
            st.metric(
                "Kích thước", f"{results['models']['bi-encoder']['size_mb']:.1f} MB"
            )
        else:
            st.error("❌ Bi-Encoder")

    with col2:
        if results.get("models", {}).get("cross-encoder", {}).get("exists"):
            st.success("✅ Cross-Encoder")
            st.metric(
                "Kích thước", f"{results['models']['cross-encoder']['size_mb']:.1f} MB"
            )
        else:
            st.error("❌ Cross-Encoder")

    with col3:
        if results.get("models", {}).get("light-reranker", {}).get("exists"):
            st.success("✅ Light Reranker")
            st.metric(
                "Kích thước", f"{results['models']['light-reranker']['size_mb']:.1f} MB"
            )
        else:
            st.error("❌ Light Reranker")

    # Display evaluation metrics
    if results.get("evaluation"):
        eval_data = results["evaluation"]

        st.markdown("## 📈 Kết quả đánh giá chi tiết")

        # Retrieval metrics
        if eval_data.get("summary", {}).get("retrieval_metrics"):
            st.markdown("### 🎯 Tầng 1 - Bi-Encoder Retrieval")
            retrieval_metrics = eval_data["summary"]["retrieval_metrics"]

            # Create metrics table
            retrieval_data = []
            for k, metrics in retrieval_metrics.items():
                if isinstance(metrics, dict):
                    retrieval_data.append(
                        {
                            "Top-K": k,
                            "Precision": f"{metrics.get('precision', 0):.4f}",
                            "Recall": f"{metrics.get('recall', 0):.4f}",
                            "F1": f"{metrics.get('f1', 0):.4f}",
                        }
                    )

            if retrieval_data:
                df_retrieval = pd.DataFrame(retrieval_data)
                st.dataframe(df_retrieval, use_container_width=True)

        # Reranking metrics (flat format)
        if eval_data.get("summary", {}).get("reranking_metrics"):
            st.markdown("### ⚡ Tầng 3 - Cross-Encoder Reranking")
            reranking_metrics = eval_data["summary"]["reranking_metrics"]
            ks = [1, 3, 5, 10, 20, 50]
            reranking_data = [
                {
                    "Top-K": k,
                    "Precision": f"{reranking_metrics.get(f'precision@{k}', 0.0):.4f}",
                    "Recall": f"{reranking_metrics.get(f'recall@{k}', 0.0):.4f}",
                    "F1": f"{reranking_metrics.get(f'f1@{k}', 0.0):.4f}",
                }
                for k in ks
            ]
            df_reranking = pd.DataFrame(reranking_data)
            st.dataframe(df_reranking, use_container_width=True)

        # Overall performance
        if eval_data.get("summary", {}).get("overall_performance"):
            st.markdown("### 📊 Hiệu suất tổng thể")
            overall = eval_data["summary"]["overall_performance"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Precision@1", f"{overall.get('avg_precision@1', 0):.4f}")
            with col2:
                st.metric("Avg Recall@10", f"{overall.get('avg_recall@10', 0):.4f}")
            with col3:
                st.metric(
                    "Reranking Improvement",
                    f"{overall.get('reranking_improvement', 0):.4f}",
                )
            with col4:
                st.metric(
                    "Retrieval Quality", f"{overall.get('retrieval_quality', 0):.4f}"
                )

        # Detailed results
        if eval_data.get("detailed_results", {}).get("per_query_analysis"):
            st.markdown("### 🔍 Kết quả chi tiết từng câu hỏi")
            per_query = eval_data["detailed_results"]["per_query_analysis"]
            query_summary = []
            for query_result in per_query[:10]:  # Show first 10 queries
                query_summary.append(
                    {
                        "Query ID": query_result.get("query_id", ""),
                        "Query": (query_result.get("query", "") or "")[:50] + "...",
                        "Retrieval F1": f"{query_result.get('retrieval_results', {}).get('f1', 0):.4f}",
                        "Reranking F1": f"{query_result.get('reranking_results', {}).get('f1', 0):.4f}",
                        "Improvement": f"{query_result.get('improvement', {}).get('f1_improvement', 0):.4f}",
                    }
                )
            if query_summary:
                df_queries = pd.DataFrame(query_summary)
                st.dataframe(df_queries, use_container_width=True)

    else:
        st.warning("📝 Chưa có dữ liệu đánh giá. Hãy chạy pipeline training trước.")

elif page == "ℹ️ Thông tin hệ thống":
    st.title("ℹ️ Thông tin hệ thống")
    st.markdown("Thông tin chi tiết về kiến trúc và cấu hình hệ thống")

    st.markdown("## 🏗️ Kiến trúc hệ thống")
    st.markdown(
        """
    **Legal QA System** sử dụng kiến trúc 3 tầng tối ưu:
    
    ### 🎯 Tầng 1 - Bi-Encoder Retrieval
    - **Mục đích:** Tìm kiếm nhanh ứng viên ban đầu
    - **Model:** Sentence Transformers (PhoBERT-based)
    - **Công nghệ:** FAISS Vector Search
    - **Hiệu suất:** ~1000+ documents/giây
    
    ### ⚡ Tầng 2 - Light Reranker  
    - **Mục đích:** Lọc xuống ứng viên chất lượng cao
    - **Model:** Light Cross-Encoder
    - **Hiệu suất:** ~100+ documents/giây
    
    ### 🎯 Tầng 3 - Cross-Encoder Reranking
    - **Mục đích:** Xếp hạng chính xác kết quả cuối cùng
    - **Model:** Heavy Cross-Encoder (PhoBERT-Law)
    - **Hiệu suất:** ~10+ documents/giây
    """
    )

    st.markdown("## 📊 Cấu hình hiện tại")

    # Load config info
    config_info = {
        "Bi-Encoder": {
            "Model": config.BI_ENCODER_MODEL_NAME,
            "Batch Size": config.BI_ENCODER_BATCH_SIZE,
            "Learning Rate": config.BI_ENCODER_LR,
            "Epochs": config.BI_ENCODER_EPOCHS,
            "Warmup Ratio": config.BI_ENCODER_WARMUP_RATIO,
            "Eval Steps": config.BI_ENCODER_EVAL_STEPS,
            "Gradient Accumulation": config.BI_ENCODER_GRADIENT_ACCUMULATION_STEPS,
            "DataLoader Workers": config.BI_ENCODER_DATALOADER_NUM_WORKERS,
            "Pin Memory": config.BI_ENCODER_DATALOADER_PIN_MEMORY,
            "Prefetch Factor": config.BI_ENCODER_DATALOADER_PREFETCH_FACTOR,
            "Early Stopping Patience": config.BI_ENCODER_EARLY_STOPPING_PATIENCE,
        },
        "Cross-Encoder": {
            "Model": config.CROSS_ENCODER_MODEL_NAME,
            "Batch Size": config.CROSS_ENCODER_BATCH_SIZE,
            "Learning Rate": config.CROSS_ENCODER_LR,
            "Epochs": config.CROSS_ENCODER_EPOCHS,
            "Max Length": config.CROSS_ENCODER_MAX_LENGTH,
            "Warmup Steps": config.CROSS_ENCODER_WARMUP_RATIO,
            "Eval Steps": config.CROSS_ENCODER_EVAL_STEPS,
            "Gradient Accumulation": config.CROSS_ENCODER_GRADIENT_ACCUMULATION_STEPS,
            "DataLoader Workers": config.CROSS_ENCODER_DATALOADER_NUM_WORKERS,
            "Pin Memory": config.CROSS_ENCODER_DATALOADER_PIN_MEMORY,
            "Prefetch Factor": config.CROSS_ENCODER_DATALOADER_PREFETCH_FACTOR,
            "Early Stopping Patience": config.CROSS_ENCODER_EARLY_STOPPING_PATIENCE,
        },
        "Light Reranker": {
            "Model": config.LIGHT_RERANKER_MODEL_NAME,
            "Batch Size": config.LIGHT_RERANKER_BATCH_SIZE,
            "Learning Rate": config.LIGHT_RERANKER_LR,
            "Epochs": config.LIGHT_RERANKER_EPOCHS,
            "Max Length": config.LIGHT_RERANKER_MAX_LENGTH,
            "Warmup Ratio": config.LIGHT_RERANKER_WARMUP_RATIO,
            "Eval Steps": config.LIGHT_RERANKER_EVAL_STEPS,
            "Gradient Accumulation": config.LIGHT_RERANKER_GRADIENT_ACCUMULATION_STEPS,
            "DataLoader Workers": config.LIGHT_RERANKER_DATALOADER_NUM_WORKERS,
            "Early Stopping Patience": config.LIGHT_RERANKER_EARLY_STOPPING_PATIENCE,
        },
    }

    for model_type, config_data in config_info.items():
        with st.expander(f"⚙️ {model_type} Configuration"):
            for key, value in config_data.items():
                st.text(f"{key}: {value}")

    st.markdown("## 🚀 Tính năng nâng cao")
    st.markdown(
        """
    - **🎲 Random Questions:** Tự động tạo câu hỏi ngẫu nhiên từ dataset
    - **⚙️ Dynamic Parameters:** Tự động tính toán tham số tối ưu cho từng tầng
    - **📊 Performance Monitoring:** Theo dõi thời gian xử lý và hiệu suất
    - **🔄 Caching:** Cache models và data để tăng tốc độ
    - **💾 Memory Management:** Quản lý bộ nhớ thông minh
    """
    )
