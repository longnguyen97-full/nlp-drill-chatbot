import streamlit as st
import sys
import os

# Them thu muc goc vao path de import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from core.pipeline import LegalQAPipeline


@st.cache_resource
def load_pipeline():
    """Tai va cache pipeline de tranh load lai moi lan tuong tac."""
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


st.set_page_config(page_title="He thong Hoi-Dap Phap luat", layout="wide")

st.title("He thong Hoi-Dap Phap luat Viet Nam")
st.markdown(
    "Nhap mot cau hoi ve phap luat va he thong se co gang tim nhung dieu luat lien quan nhat."
)

pipeline = load_pipeline()

if pipeline:
    query = st.text_input(
        "Nhap cau hoi cua ban:", "Nguoi lao dong duoc nghi phep bao nhieu ngay?"
    )

    with st.sidebar:
        st.header("Tuy chon nang cao")
        top_k_retrieval = st.slider(
            "So luong ung vien (Tang 1)",
            min_value=10,
            max_value=500,
            value=config.TOP_K_RETRIEVAL,
            step=10,
        )
        top_k_final = st.slider(
            "So ket qua cuoi cung (Tang 2)",
            min_value=1,
            max_value=20,
            value=config.TOP_K_FINAL,
            step=1,
        )

    if st.button("Tim kiem", type="primary"):
        if query:
            with st.spinner("Dang tim kiem va xep hang cac dieu luat..."):
                results = pipeline.predict(
                    query=query,
                    top_k_retrieval=top_k_retrieval,
                    top_k_final=top_k_final,
                )

            st.success("Hoan thanh!")

            if not results:
                st.warning("Khong tim thay ket qua phu hop.")
            else:
                st.subheader(f"Top {len(results)} ket qua lien quan nhat:")
                for i, res in enumerate(results):
                    with st.expander(
                        f"**Ket qua {i+1}: Dieu {res['aid']}** (Diem: {res['rerank_score']:.4f})"
                    ):
                        st.markdown(f"**Noi dung:**")
                        st.write(res["content"])
                        st.markdown(f"---")
                        st.markdown(
                            f"**Diem Retrieval (Tang 1):** {res['retrieval_score']:.4f}"
                        )
                        st.markdown(
                            f"**Diem Re-rank (Tang 2):** {res['rerank_score']:.4f}"
                        )
        else:
            st.warning("Vui long nhap mot cau hoi.")
else:
    st.header("Pipeline chua san sang")
    st.info(
        "Vui long kiem tra terminal de biet ly do loi va dam bao cac mo hinh da duoc chuan bi."
    )
