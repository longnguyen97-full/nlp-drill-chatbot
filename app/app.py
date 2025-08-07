import streamlit as st
import sys
import os

# Them thu muc goc vao path de import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from core.pipeline import LegalQAPipeline


@st.cache_resource
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

        pipeline = LegalQAPipeline()
        if not pipeline.is_ready:
            st.error(
                "Loi khoi tao Pipeline. Vui long kiem tra logs o terminal de biet chi tiet."
            )
            st.warning(
                "Hay chac chan rang ban da huan luyen cac mo hinh va dat chung vao thu muc 'models', sau do chay 'scripts/04_build_faiss_index.py'."
            )
            return None
        return pipeline
    except Exception as e:
        st.error(f"Lỗi khởi tạo pipeline: {e}")
        st.info("Thử khởi động lại app hoặc kiểm tra logs")
        return None


st.set_page_config(page_title="Hệ thống Hỏi-Đáp Pháp luật", layout="wide")

st.title("🏛️ Hệ thống Hỏi-Đáp Pháp luật Việt Nam")
st.markdown(
    "Nhập một câu hỏi về pháp luật và hệ thống sẽ cố gắng tìm những điều luật liên quan nhất."
)

# Add architecture info
with st.expander("ℹ️ Kiến trúc hệ thống (3 tầng)"):
    st.markdown(
        """
    **🎯 Tầng 1 - Bi-Encoder Retrieval:** Tìm kiếm nhanh 100-500 ứng viên  
    **⚡ Tầng 2 - Light Reranker:** Lọc xuống 50 ứng viên chất lượng cao  
    **🎯 Tầng 3 - Cross-Encoder Reranking:** Xếp hạng chính xác 5 kết quả cuối cùng
    """
    )

# Add device selection option
with st.sidebar:
    st.header("⚙️ Cài đặt hệ thống")
    force_cpu = st.checkbox("🖥️ Force sử dụng CPU (nếu có lỗi CUDA)", value=False)

pipeline = load_pipeline(force_cpu=force_cpu)

if pipeline:
    query = st.text_input(
        "Nhập câu hỏi của bạn:", "Người lao động được nghỉ phép bao nhiêu ngày?"
    )

    with st.sidebar:
        st.header("🔧 Tùy chọn nâng cao")
        top_k_retrieval = st.slider(
            "Số lượng ứng viên (Tầng 1)",
            min_value=10,
            max_value=500,
            value=config.TOP_K_RETRIEVAL,
            step=10,
        )
        top_k_final = st.slider(
            "Số kết quả cuối cùng (Tầng 3)",
            min_value=1,
            max_value=20,
            value=config.TOP_K_FINAL,
            step=1,
        )

    if st.button("🔍 Tìm kiếm", type="primary"):
        if query:
            with st.spinner("🔄 Đang tìm kiếm và xếp hạng các điều luật..."):
                results = pipeline.predict(
                    query=query,
                    top_k_retrieval=top_k_retrieval,
                    top_k_final=top_k_final,
                )

            st.success("✅ Hoàn thành!")

            if not results:
                st.warning("⚠️ Không tìm thấy kết quả phù hợp.")
            else:
                st.subheader(f"📋 Top {len(results)} kết quả liên quan nhất:")
                for i, res in enumerate(results):
                    with st.expander(
                        f"**Kết quả {i+1}: Điều {res['aid']}** (Điểm: {res['rerank_score']:.4f})"
                    ):
                        st.markdown(f"**📄 Nội dung:**")
                        st.write(res["content"])
                        st.markdown(f"---")
                        st.markdown(
                            f"**🎯 Điểm Retrieval (Tầng 1 - Bi-Encoder):** {res['retrieval_score']:.4f}"
                        )
                        st.markdown(
                            f"**⚡ Điểm Re-rank (Tầng 3 - Cross-Encoder):** {res['rerank_score']:.4f}"
                        )
                        st.markdown(
                            f"**📊 Tổng điểm:** {(res['retrieval_score'] + res['rerank_score']) / 2:.4f}"
                        )
        else:
            st.warning("⚠️ Vui lòng nhập một câu hỏi.")
else:
    st.header("🚨 Pipeline chưa sẵn sàng")
    st.info(
        "⚠️ Vui lòng kiểm tra terminal để biết lý do lỗi và đảm bảo các mô hình đã được chuẩn bị."
    )
