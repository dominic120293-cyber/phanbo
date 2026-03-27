import streamlit as st
import io
import time
import threading
import pandas as pd
from ALLOCATION import run_optimization

# ==========================================
# CẤU HÌNH GIAO DIỆN WEB
# ==========================================
st.set_page_config(page_title="DISTRIBUTION CONTAINER", page_icon="🚢", layout="centered")
st.markdown("""
    <style>
        /* ── Nền đen toàn trang ── */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stApp { background-color: #000000 !important; }

        .main > div { max-width: 800px; margin: 0 auto; }

        /* ── Chữ trắng ── */
        h1, h2, h3, p, label, div, span,
        .stMarkdown, [data-testid="stMarkdownContainer"] {
            color: #e8e8e8 !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            background-color: #111111 !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 8px;
        }

        /* ── Nút bấm thường ── */
        .stButton > button {
            background-color: #0d1117 !important;
            color: #58a6ff !important;
            border: 1px solid #2a4a7f !important;
            border-radius: 8px !important;
        }
        .stButton > button:hover {
            background-color: #161b22 !important;
            border-color: #4a8adf !important;
        }

        /* ── Info / Success / Error boxes ── */
        [data-testid="stAlert"] {
            background-color: #0d1117 !important;
            border-color: #30363d !important;
        }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
            background-color: #0d1117 !important;
            border: 1px solid #21262d !important;
            border-radius: 8px;
            padding: 12px;
        }

        /* ── Download button ── */
        [data-testid="stDownloadButton"] > button {
            background-color: #0f3460 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
        }
        [data-testid="stDownloadButton"] > button:hover {
            background-color: #1a4f8a !important;
        }

        /* ── Timer box ── */
        .timer-box {
            display: flex;
            align-items: center;
            gap: 12px;
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 14px 20px;
            margin-top: 8px;
        }
        .timer-label  { color: #8b949e !important; font-size: 15px; font-weight: 500; }
        .timer-value  { font-size: 28px; font-weight: 700; color: #58a6ff !important;
                        font-variant-numeric: tabular-nums; letter-spacing: 2px; }
        .spinner-msg  { color: #8b949e !important; font-size: 15px; margin-bottom: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚢 HỆ THỐNG PHÂN BỔ TỐI ƯU TỰ ĐỘNG")
st.markdown("Tải file dữ liệu đầu vào (Excel) để hệ thống chạy thuật toán và trả về kết quả.")

# ==========================================
# KHU VỰC TẢI FILE LÊN (IMPORT)
# ==========================================
uploaded_file = st.file_uploader("📂 Tải lên file Excel Input của bạn", type=["xlsx", "xls"])

if uploaded_file is not None:
    st.info("Đã nhận file. Bấm nút bên dưới để tiến hành tính toán.")

    if st.button("🚀 CHẠY THUẬT TOÁN TỐI ƯU", use_container_width=True):

        file_bytes = io.BytesIO(uploaded_file.read())
        result_holder = {"done": False, "result": None, "error": None}

        def run_in_thread():
            try:
                result_holder["result"] = run_optimization(file_bytes)
            except Exception as e:
                result_holder["error"] = e
            finally:
                result_holder["done"] = True

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()
        start_time = time.time()

        st.markdown('<p class="spinner-msg">⏳ Hệ thống đang tính toán và xuất file... Vui lòng đợi!</p>',
                    unsafe_allow_html=True)
        timer_placeholder = st.empty()

        while not result_holder["done"]:
            elapsed = int(time.time() - start_time)
            mm, ss = divmod(elapsed, 60)
            timer_placeholder.markdown(
                f"""<div class="timer-box">
                      <span class="timer-label">⏱ Thời gian tính toán:</span>
                      <span class="timer-value">{mm:02d}:{ss:02d}</span>
                    </div>""",
                unsafe_allow_html=True
            )
            time.sleep(0.5)

        elapsed = int(time.time() - start_time)
        mm, ss = divmod(elapsed, 60)
        timer_placeholder.markdown(
            f"""<div class="timer-box" style="border-color:#1a3a2a; background:#0a1a0f;">
                  <span class="timer-label">✅ Hoàn tất sau:</span>
                  <span class="timer-value" style="color:#3fb950 !important;">{mm:02d}:{ss:02d}</span>
                </div>""",
            unsafe_allow_html=True
        )

        if result_holder["error"]:
            st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán.\n\nChi tiết lỗi: {result_holder['error']}")
        else:
            excel_buffer, total_rows, objective_value = result_holder["result"]

            st.success("🎉 Tính toán hoàn tất!")

            col1, col2 = st.columns(2)
            col1.metric("Tổng số dòng phân bổ", f"{total_rows} dòng")
            col2.metric("Số lượng Clash (giá trị mục tiêu)", f"{objective_value}", delta_color="inverse")

            st.download_button(
                label="📥 TẢI FILE KẾT QUẢ",
                data=excel_buffer,
                file_name="PHAN_BO.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
