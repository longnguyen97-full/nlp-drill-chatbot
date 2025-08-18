#!/usr/bin/env python3
"""
LawBot - Main Application
=========================

Main Streamlit application with a modular page architecture.
This is the single entry point for the user interface.
"""

import streamlit as st
import sys
from pathlib import Path
import os
import time

# Import pages with proper error handling
try:
    # Use relative imports when running from app directory
    from .pages import search, analysis, system

    pages_loaded = True

except ImportError as e:
    # Fallback for direct execution
    try:
        # Check if LAWBOT_PROJECT_ROOT is set (from run_app.py)
        if os.environ.get("LAWBOT_PROJECT_ROOT"):
            project_root = os.environ["LAWBOT_PROJECT_ROOT"]
            sys.path.insert(0, project_root)
        else:
            # Add parent directory to path for direct execution
            sys.path.append(str(Path(__file__).parent.parent))

        from app.pages import search, analysis, system

        pages_loaded = True
    except ImportError as e2:
        st.error(
            f"**Lỗi Import Module:** Không thể tải các trang của ứng dụng. Lỗi: `{e2}`\n\n"
            f"Vui lòng đảm bảo bạn đang chạy ứng dụng từ thư mục gốc của dự án và đã cài đặt tất cả các gói phụ thuộc.\n\n"
            f"**Hướng dẫn khắc phục:**\n"
            f"1. Chạy từ thư mục gốc: `python run_app.py`\n"
            f"2. Kiểm tra cài đặt: `pip install -r requirements.txt`\n"
            f"3. Kiểm tra cấu trúc thư mục app/pages/"
        )
        pages_loaded = False
        st.stop()


# Lazy load pipeline to avoid import errors
def get_pipeline():
    """Lazy load pipeline only when needed."""
    try:
        from core.pipeline import LegalQAPipeline

        return LegalQAPipeline()
    except Exception as e:
        st.warning(f"⚠️ Pipeline not available: {e}")
        return None


def render_app():
    """Render the main application interface."""
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Hệ thống Hỏi-Đáp Pháp luật",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Hide default Streamlit navigation ---
    hide_default_navigation = """
    <style>
    /* Hide default Streamlit navigation elements - minimal approach */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide built-in multipage sidebar nav */
    section[data-testid="stSidebarNav"],
    div[data-testid="stSidebarNav"] {
        display: none !important;
    } 
    
    /* Clean page transitions - prevent element bleeding */
    .stApp > div {
        opacity: 1 !important;
        transition: opacity 0.2s ease-in-out;
    }
    
    /* Ensure no lingering elements from previous pages */
    .stApp > div[style*="opacity"] {
        opacity: 1 !important;
    }
    </style>
    """
    st.markdown(hide_default_navigation, unsafe_allow_html=True)

    # --- Sidebar Navigation ---
    st.sidebar.title("🎯 Hệ thống QA Pháp luật")

    if pages_loaded:
        page_options = {
            "🔍 Tìm kiếm & Hỏi đáp": search.render_search_page,
            "📊 Phân tích & Báo cáo": analysis.render_analysis_page,
            "🔧 Trạng thái": system.main,
        }

        selected_page_title = st.sidebar.selectbox(
            "Chọn trang:",
            list(page_options.keys()),
            index=0,
            key="main_page_selector",
        )

        # --- Main Content ---
        if selected_page_title in page_options:
            # Enhanced page state management to prevent element bleeding
            if "current_page" not in st.session_state:
                st.session_state.current_page = selected_page_title
                st.session_state.page_load_time = time.time()

            # Clear page-specific session states when switching pages
            if st.session_state.current_page != selected_page_title:
                # Clear all page-specific states to prevent element bleeding
                keys_to_clear = [
                    "analysis_page_loaded",
                    "search_page_loaded",
                    "system_page_loaded",
                    "comprehensive_eval_results",
                    "eval_loading",
                    "switch_to_tab2",
                    "pipeline_loaded",
                    "search_results",
                    "search_query",
                ]

                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

                # Update current page and force clean render
                st.session_state.current_page = selected_page_title
                st.session_state.page_load_time = time.time()

                # Force clean page transition
                st.rerun()

            # Render the selected page with clean state
            page_options[selected_page_title]()

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**LawBot v8.3** - Hệ thống QA Pháp luật thông minh")
    st.sidebar.markdown("Powered by AI & Legal Knowledge Base")


def main():
    """Main application entry point."""
    render_app()


if __name__ == "__main__":
    main()
