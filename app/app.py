import streamlit as st
import sys
import os
import json
import random
import time
import gc
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Them thu muc goc vao path de import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from core.pipeline import LegalQAPipeline


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_random_questions() -> Tuple[List[str], List[str]]:
    """
    Load random questions from training and test datasets
    Returns: (training_questions, test_questions)
    """
    training_questions = []
    test_questions = []

    try:
        # Load training questions
        train_file = os.path.join(config.DATA_DIR, "train.json")
        if os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
                training_questions = [
                    item.get("question", "")
                    for item in train_data
                    if item.get("question")
                ]

        # Load test questions
        test_file = os.path.join(config.DATA_DIR, "public_test.json")
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

# Main title
st.title("🏛️ Hệ thống Hỏi-Đáp Pháp luật Việt Nam")
st.markdown(
    "Nhập một câu hỏi về pháp luật và hệ thống sẽ cố gắng tìm những điều luật liên quan nhất."
)

# Load random questions
training_questions, test_questions = load_random_questions()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Cài đặt hệ thống")

    # Device selection
    force_cpu = st.checkbox("🖥️ Force sử dụng CPU (nếu có lỗi CUDA)", value=False)

    # Performance settings
    st.subheader("🚀 Cài đặt hiệu suất")
    search_aggressiveness = st.selectbox(
        "Mức độ tìm kiếm",
        options=["conservative", "balanced", "aggressive"],
        index=1,
        help="Conservative: Nhanh hơn, ít kết quả hơn. Aggressive: Chậm hơn, nhiều kết quả hơn",
    )

    # Results customization
    st.subheader("🔧 Tùy chọn kết quả")
    final_results_count = st.slider(
        "Số kết quả cuối cùng",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Số lượng điều luật sẽ hiển thị trong kết quả cuối cùng",
    )

    # Random question feature
    st.subheader("🎲 Câu hỏi ngẫu nhiên")
    if training_questions or test_questions:
        question_source = st.selectbox(
            "Chọn nguồn câu hỏi",
            options=["Training Dataset", "Test Dataset", "Cả hai"],
            help="Câu hỏi từ training dataset thường đa dạng hơn",
        )

        if st.button("🎲 Lấy câu hỏi ngẫu nhiên"):
            if question_source == "Training Dataset" and training_questions:
                random_q = get_random_question(training_questions, "training")
                st.session_state.random_question = random_q
            elif question_source == "Test Dataset" and test_questions:
                random_q = get_random_question(test_questions, "test")
                st.session_state.random_question = random_q
            elif question_source == "Cả hai":
                all_questions = training_questions + test_questions
                random_q = get_random_question(all_questions, "combined")
                st.session_state.random_question = random_q
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
    # Query input with random question support
    default_question = "Người lao động được nghỉ phép bao nhiêu ngày?"

    # Use random question if available
    if hasattr(st.session_state, "random_question"):
        default_question = st.session_state.random_question
        # Clear the random question after use
        del st.session_state.random_question

    query = st.text_input(
        "Nhập câu hỏi của bạn:",
        value=default_question,
        help="Nhập câu hỏi về pháp luật hoặc sử dụng tính năng câu hỏi ngẫu nhiên",
    )

    if st.button("🔍 Tìm kiếm", type="primary"):
        if query:
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

            with st.spinner("🔄 Đang tìm kiếm và xếp hạng các điều luật..."):
                try:
                    results = pipeline.predict(
                        query=query,
                        top_k_retrieval=params["top_k_retrieval"],
                        top_k_final=params["top_k_final"],
                        top_k_light_reranking=params["top_k_light_reranking"],
                    )

                    end_time = time.time()

                    # Display performance metrics
                    display_performance_metrics(
                        start_time, end_time, len(results) if results else 0
                    )

                except Exception as e:
                    st.error(f"❌ Lỗi trong quá trình tìm kiếm: {e}")
                    st.info("💡 Thử giảm số lượng kết quả hoặc chuyển sang CPU mode")
                    results = None

            if results is not None:
                st.success("✅ Hoàn thành!")

            if not results:
                st.warning("⚠️ Không tìm thấy kết quả phù hợp.")
                st.info("💡 Thử sử dụng từ khóa khác hoặc tăng mức độ tìm kiếm")
            else:
                st.subheader(f"📋 Top {len(results)} kết quả liên quan nhất:")

                # Create tabs for different views
                tab1, tab2 = st.tabs(["📄 Xem chi tiết", "📊 So sánh điểm số"])

                with tab1:
                    for i, res in enumerate(results):
                        with st.expander(
                            f"**Kết quả {i+1}: Điều {res['aid']}** (Điểm: {res['rerank_score']:.4f})"
                        ):
                            st.markdown(f"**📄 Nội dung:**")
                            st.write(res["content"])
                            st.markdown(f"---")
                            st.markdown(
                                f"**🎯 Điểm Retrieval (Tầng 1):** {res['retrieval_score']:.4f}"
                            )
                            st.markdown(
                                f"**⚡ Điểm Re-rank (Tầng 3):** {res['rerank_score']:.4f}"
                            )
                            st.markdown(
                                f"**📊 Tổng điểm:** {(res['retrieval_score'] + res['rerank_score']) / 2:.4f}"
                            )

                with tab2:
                    # Create comparison chart
                    if results:
                        df = pd.DataFrame(
                            [
                                {
                                    "Kết quả": f"Điều {res['aid']}",
                                    "Retrieval Score": res["retrieval_score"],
                                    "Rerank Score": res["rerank_score"],
                                    "Tổng điểm": (
                                        res["retrieval_score"] + res["rerank_score"]
                                    )
                                    / 2,
                                }
                                for res in results
                            ]
                        )

                        st.bar_chart(
                            df.set_index("Kết quả")[["Retrieval Score", "Rerank Score"]]
                        )

                        # Display summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Điểm cao nhất",
                                f"{max([r['rerank_score'] for r in results]):.4f}",
                            )
                        with col2:
                            st.metric(
                                "Điểm thấp nhất",
                                f"{min([r['rerank_score'] for r in results]):.4f}",
                            )
                        with col3:
                            avg_score = sum([r["rerank_score"] for r in results]) / len(
                                results
                            )
                            st.metric("Điểm trung bình", f"{avg_score:.4f}")
        else:
            st.warning("⚠️ Vui lòng nhập một câu hỏi.")
else:
    st.header("🚨 Pipeline chưa sẵn sàng")
    st.info(
        "⚠️ Vui lòng kiểm tra terminal để biết lý do lỗi và đảm bảo các mô hình đã được chuẩn bị."
    )
