#!/usr/bin/env python3
"""
System Status & Configuration Page - LawBot v8.3
===============================================

Simplified system status monitoring with essential information only.
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from core.utils.system_check import get_model_status, get_faiss_index_status
    from core.utils.io import load_json
    from core.utils.logging_manager import get_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Setup logger
logger = get_logger("system_page")


def load_system_data():
    """Load essential system data."""
    try:
        # Load model status
        model_status = get_model_status()

        # Load FAISS status
        faiss_status = get_faiss_index_status()

        # Load basic evaluation data if available
        evaluation_data = None
        evaluation_dir = Path("evaluation")
        if evaluation_dir.exists():
            comp_files = list(evaluation_dir.glob("comprehensive_evaluation_*.json"))
            if comp_files:
                latest_comp = max(comp_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_comp, "r", encoding="utf-8") as f:
                        evaluation_data = json.load(f)
                except:
                    pass

        return {
            "model_status": model_status,
            "faiss_status": faiss_status,
            "evaluation_data": evaluation_data,
        }
    except Exception as e:
        logger.error(f"Failed to load system data: {e}")
        return None


def create_system_health_dashboard(system_data):
    """Create simplified system health dashboard."""
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
            1 for model in model_status.values() if model.get("status") == "ready"
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
        faiss_ready = (
            "✅ Ready" if faiss_status.get("status") == "ready" else "❌ Not Ready"
        )
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

    fig = go.Figure(
        go.Indicator(
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
                    {"range": [80, 100], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def create_tier_configuration_info(system_data):
    """Create tier configuration information display."""
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
            "status_key": "bi_encoder",
        },
        "Light Reranker (Tầng 2)": {
            "model": model_status.get("light_reranker", {}),
            "description": "Light reranking model sử dụng PhoBERT",
            "status_key": "light_reranker",
        },
        "Cross-Encoder (Tầng 3)": {
            "model": model_status.get("cross_encoder", {}),
            "description": "Cross-encoder model kết hợp ensemble",
            "status_key": "cross_encoder",
        },
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
                    model_path = model_info.get(
                        "path", "N/A"
                    )  # Changed from "model_path" to "path"
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


def create_faiss_index_status(system_data):
    """Create FAISS Index Status display."""
    st.subheader("🔍 FAISS Index Status")

    if not system_data:
        st.warning("⚠️ Không thể load thông tin FAISS Index")
        return

    faiss_status = system_data.get("faiss_status", {})

    if faiss_status:
        # Status overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = faiss_status.get("status", "unknown")
            if status == "ready":
                st.success("✅ Index Status")
                st.metric("Status", "Ready")
            else:
                st.error("❌ Index Status")
                st.metric("Status", "Not Ready")

        with col2:
            index_size = faiss_status.get("index_size", 0)
            st.metric("Index Size", f"{index_size:,} vectors")

        with col3:
            file_count = faiss_status.get("file_count", 0)
            st.metric("Index Files", file_count)

        with col4:
            size_mb = faiss_status.get("size_mb", 0)
            st.metric("Disk Size", f"{size_mb:.1f} MB")

        # Detailed information
        with st.expander("📋 Chi tiết FAISS Index", expanded=False):
            if faiss_status.get("files"):
                st.write("**Index Files:**")
                for file in faiss_status["files"]:
                    st.code(file)

            # Index health check
            if faiss_status.get("status") == "ready":
                st.success("✅ FAISS Index hoạt động bình thường")
                if faiss_status.get("index_size", 0) > 0:
                    st.info(
                        f"Index chứa {faiss_status['index_size']:,} vectors - đủ để tìm kiếm"
                    )
                else:
                    st.warning("⚠️ Index không có vectors nào")
            else:
                st.error("❌ FAISS Index có vấn đề - cần kiểm tra")
    else:
        st.warning("⚠️ Không có thông tin FAISS Index")


def create_system_summary(system_data):
    """Create system summary with recommendations."""
    st.subheader("📊 Tóm tắt Hệ thống & Khuyến nghị")

    if not system_data:
        st.warning("⚠️ Không thể tạo tóm tắt hệ thống")
        return

    # Calculate system health
    model_status = system_data.get("model_status", {})
    faiss_status = system_data.get("faiss_status", {})

    models_ready = sum(
        1 for model in model_status.values() if model.get("status") == "ready"
    )
    total_models = len(model_status)
    faiss_ready = faiss_status.get("status") == "ready"

    # Overall assessment
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Đánh giá Tổng quan")

        if models_ready == total_models and faiss_ready:
            st.success("🎉 Hệ thống hoạt động hoàn hảo!")
            st.info("Tất cả models và FAISS index đều sẵn sàng")
        elif models_ready >= total_models * 0.7 and faiss_ready:
            st.warning("⚠️ Hệ thống hoạt động tốt")
            st.info("Một số models cần kiểm tra")
        else:
            st.error("🔴 Hệ thống cần được kiểm tra")
            st.info("Nhiều components có vấn đề")

    with col2:
        st.subheader("📈 Thống kê")
        st.metric("Models Ready", f"{models_ready}/{total_models}")
        st.metric("FAISS Status", "✅ Ready" if faiss_ready else "❌ Not Ready")

        if total_models > 0:
            health_percentage = (models_ready / total_models) * 100
            st.metric("System Health", f"{health_percentage:.0f}%")

    # Recommendations
    st.subheader("💡 Khuyến nghị")

    if models_ready < total_models:
        st.warning("⚠️ **Models cần kiểm tra:**")
        for name, status in model_status.items():
            if status.get("status") != "ready":
                st.write(f"- {name}: {status.get('status', 'unknown')}")

    if not faiss_ready:
        st.error("🔴 **FAISS Index cần kiểm tra:**")
        st.write("- Kiểm tra file index có tồn tại không")
        st.write("- Kiểm tra quyền truy cập file")

    if models_ready == total_models and faiss_ready:
        st.success("✅ **Hệ thống sẵn sàng:**")
        st.write("- Tất cả components hoạt động bình thường")
        st.write("- Có thể thực hiện tìm kiếm và phân tích")


def main():
    """Main system page function."""
    # Clean page state and ensure no element bleeding
    if "system_page_loaded" not in st.session_state:
        st.session_state.system_page_loaded = True

    # Page cleanup CSS - ensure clean rendering without conflicts
    cleanup_css = """
    <style>
    /* No conflicts with app.py CSS */
    </style>
    """
    st.markdown(cleanup_css, unsafe_allow_html=True)

    st.title("🔧 Trạng thái Hệ thống & Cấu hình")
    st.markdown("Giám sát trạng thái hệ thống và thông tin cấu hình các tầng")

    # Load system data
    with st.spinner("🔄 Đang tải thông tin hệ thống..."):
        system_data = load_system_data()

    if system_data:
        # Create main sections
        create_system_health_dashboard(system_data)
        create_tier_configuration_info(system_data)
        create_faiss_index_status(system_data)
        create_system_summary(system_data)

        # Debug section (collapsible)
        with st.expander("🔍 Debug - Raw System Data", expanded=False):
            st.json(system_data)
    else:
        st.error("❌ Không thể load thông tin hệ thống")
        st.info("Vui lòng kiểm tra logs và thử lại")


if __name__ == "__main__":
    main()
