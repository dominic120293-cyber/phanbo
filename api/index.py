import io
import sys
import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import tempfile

# Thêm đường dẫn thư mục cha để import ALLOCATION.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ALLOCATION import run_optimization

app = FastAPI(title="HỆ THỐNG PHÂN BỔ TỐI ƯU")

# ------------------- HTML giao diện (Dark Theme) -------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ thống phân bổ tối ưu tự động</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            background: #0b0e14;
            color: #e8edf5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            margin: 0;
        }
        .container {
            background: #171e26;
            border-radius: 24px;
            padding: 40px 48px;
            max-width: 640px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.7);
            border: 1px solid #2a343f;
            transition: all 0.2s;
        }
        h1 {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #8ab4f8, #a8d8ea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            text-align: center;
        }
        .sub {
            text-align: center;
            color: #8899aa;
            font-size: 14px;
            margin-bottom: 30px;
            border-bottom: 1px solid #2a343f;
            padding-bottom: 20px;
        }
        .upload-area {
            background: #1f2937;
            border: 2px dashed #3b4a5a;
            border-radius: 16px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.3s, background 0.3s;
            margin-bottom: 24px;
        }
        .upload-area:hover {
            border-color: #6a8cff;
            background: #253240;
        }
        .upload-area.dragover {
            border-color: #6a8cff;
            background: #2a3a4a;
        }
        .upload-area input[type="file"] {
            display: none;
        }
        .upload-icon {
            font-size: 48px;
            line-height: 1;
            margin-bottom: 8px;
        }
        .upload-text {
            font-size: 16px;
            color: #b0c4de;
        }
        .upload-text strong {
            color: #8ab4f8;
        }
        .file-name {
            margin-top: 12px;
            font-size: 14px;
            color: #9aabbb;
            word-break: break-all;
        }
        .row {
            display: flex;
            gap: 16px;
            margin: 20px 0 24px;
        }
        .btn {
            flex: 1;
            padding: 14px 20px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.2s, background 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn:active { transform: scale(0.96); }
        .btn-primary {
            background: #4c7de0;
            color: white;
            box-shadow: 0 6px 18px rgba(76, 125, 224, 0.3);
        }
        .btn-primary:hover:not(:disabled) {
            background: #5f8df0;
            box-shadow: 0 8px 24px rgba(76, 125, 224, 0.4);
        }
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            box-shadow: none;
        }
        .btn-success {
            background: #2b8c5e;
            color: white;
            box-shadow: 0 6px 18px rgba(43, 140, 94, 0.3);
        }
        .btn-success:hover:not(:disabled) {
            background: #34a06e;
            box-shadow: 0 8px 24px rgba(43, 140, 94, 0.4);
        }
        .btn-success:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .timer-box {
            background: #1f2937;
            border-radius: 12px;
            padding: 12px 16px;
            text-align: center;
            font-variant-numeric: tabular-nums;
            letter-spacing: 1px;
            margin-bottom: 16px;
            border: 1px solid #2a343f;
        }
        .timer-box .label {
            font-size: 12px;
            color: #8899aa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .timer-box .time {
            font-size: 28px;
            font-weight: 600;
            color: #8ab4f8;
            margin-top: 2px;
        }
        .status {
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 14px;
            margin-top: 12px;
            display: none;
            align-items: center;
            gap: 10px;
        }
        .status.show { display: flex; }
        .status.info { background: #1f3a4a; color: #8ab4f8; border: 1px solid #2a4a6a; }
        .status.success { background: #1a3a2a; color: #6fcf97; border: 1px solid #2a6a4a; }
        .status.error { background: #3a1a1a; color: #f28b82; border: 1px solid #6a2a2a; }
        .status .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result-info {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #b0c4de;
            margin-top: 8px;
            padding: 8px 4px;
            border-top: 1px solid #2a343f;
        }
        .result-info span {
            background: #1f2937;
            padding: 4px 12px;
            border-radius: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 12px;
            color: #556677;
        }
        /* Responsive */
        @media (max-width: 600px) {
            .container { padding: 24px 20px; }
            h1 { font-size: 22px; }
            .row { flex-direction: column; }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>⚙️ HỆ THỐNG PHÂN BỔ TỐI ƯU TỰ ĐỘNG</h1>
    <div class="sub">Tối ưu container theo MOVEHOUR &amp; WEIGHTCLASS</div>

    <!-- Vùng upload -->
    <div class="upload-area" id="dropZone">
        <div class="upload-icon">📂</div>
        <div class="upload-text"><strong>Nhấn để chọn</strong> hoặc kéo thả file <strong>Input Data.xlsx</strong></div>
        <input type="file" id="fileInput" accept=".xlsx,.xls">
        <div class="file-name" id="fileName"></div>
    </div>

    <!-- Timer -->
    <div class="timer-box">
        <div class="label">⏱️ Thời gian xử lý</div>
        <div class="time" id="timerDisplay">00:00</div>
    </div>

    <!-- Nút hành động -->
    <div class="row">
        <button class="btn btn-primary" id="runBtn" disabled>▶ CHẠY PHÂN BỔ</button>
        <button class="btn btn-success" id="downloadBtn" disabled>⬇ TẢI KẾT QUẢ</button>
    </div>

    <!-- Trạng thái -->
    <div class="status" id="statusBox">
        <div class="spinner" id="spinnerIcon"></div>
        <span id="statusText">Đang xử lý...</span>
    </div>

    <!-- Thông tin kết quả -->
    <div class="result-info" id="resultInfo" style="display:none;">
        <span>📊 Tổng dòng: <strong id="totalRows">0</strong></span>
        <span>⚡ Clashes: <strong id="totalClashes">0</strong></span>
    </div>

    <div class="footer">Phiên bản tối ưu v7 – Eviction pass</div>
</div>

<script>
    (function() {
        const fileInput = document.getElementById('fileInput');
        const dropZone = document.getElementById('dropZone');
        const fileName = document.getElementById('fileName');
        const runBtn = document.getElementById('runBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const timerDisplay = document.getElementById('timerDisplay');
        const statusBox = document.getElementById('statusBox');
        const statusText = document.getElementById('statusText');
        const spinnerIcon = document.getElementById('spinnerIcon');
        const resultInfo = document.getElementById('resultInfo');
        const totalRows = document.getElementById('totalRows');
        const totalClashes = document.getElementById('totalClashes');

        let selectedFile = null;
        let timerInterval = null;
        let startTime = null;
        let isRunning = false;

        // Cập nhật tên file
        function updateFileDisplay(file) {
            if (file) {
                fileName.textContent = '📎 ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
                runBtn.disabled = false;
            } else {
                fileName.textContent = '';
                runBtn.disabled = true;
            }
        }

        // Sự kiện chọn file
        fileInput.addEventListener('change', function(e) {
            if (this.files.length > 0) {
                selectedFile = this.files[0];
                updateFileDisplay(selectedFile);
            } else {
                selectedFile = null;
                updateFileDisplay(null);
            }
        });

        // Kéo thả
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
                    fileInput.files = e.dataTransfer.files;
                    selectedFile = file;
                    updateFileDisplay(file);
                } else {
                    showStatus('error', '⚠️ Vui lòng chọn file Excel (.xlsx / .xls)');
                }
            }
        });
        dropZone.addEventListener('click', function() {
            fileInput.click();
        });

        // Hiển thị trạng thái
        function showStatus(type, message) {
            statusBox.className = 'status show ' + type;
            statusText.textContent = message;
            if (type === 'info') {
                spinnerIcon.style.display = 'inline-block';
            } else {
                spinnerIcon.style.display = 'none';
            }
        }

        function hideStatus() {
            statusBox.className = 'status';
            spinnerIcon.style.display = 'none';
        }

        // Timer
        function startTimer() {
            startTime = Date.now();
            timerInterval = setInterval(function() {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const secs = String(elapsed % 60).padStart(2, '0');
                timerDisplay.textContent = mins + ':' + secs;
            }, 200);
        }

        function stopTimer() {
            clearInterval(timerInterval);
            timerInterval = null;
        }

        function resetTimer() {
            stopTimer();
            timerDisplay.textContent = '00:00';
            startTime = null;
        }

        // Xử lý chạy phân bổ
        runBtn.addEventListener('click', async function() {
            if (isRunning) return;
            if (!selectedFile) {
                showStatus('error', '❌ Chưa chọn file!');
                return;
            }

            // Reset UI
            hideStatus();
            resultInfo.style.display = 'none';
            downloadBtn.disabled = true;
            runBtn.disabled = true;
            isRunning = true;

            // Bắt đầu timer
            resetTimer();
            startTimer();

            // Hiển thị đang xử lý
            showStatus('info', '⏳ Đang tối ưu... vui lòng chờ');

            try {
                const formData = new FormData();
                formData.append('file', selectedFile);

                const response = await fetch('/optimize', {
                    method: 'POST',
                    body: formData
                });

                // Dừng timer
                stopTimer();
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const secs = String(elapsed % 60).padStart(2, '0');
                timerDisplay.textContent = mins + ':' + secs;

                if (!response.ok) {
                    let errMsg = 'Lỗi máy chủ';
                    try {
                        const errJson = await response.json();
                        if (errJson.detail) errMsg = errJson.detail;
                    } catch (e) {}
                    throw new Error(errMsg);
                }

                // Lấy header thông tin
                const rows = response.headers.get('X-Total-Rows') || '0';
                const clashes = response.headers.get('X-Total-Clashes') || '0';
                totalRows.textContent = rows;
                totalClashes.textContent = clashes;
                resultInfo.style.display = 'flex';

                // Tạo blob và link tải
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                downloadBtn.onclick = function() {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'Ket_qua_phan_bo.xlsx';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                };
                downloadBtn.disabled = false;

                showStatus('success', '✅ Thành công! Nhấn "TẢI KẾT QUẢ" để lưu file.');
            } catch (error) {
                showStatus('error', '❌ Lỗi: ' + error.message);
                resultInfo.style.display = 'none';
                downloadBtn.disabled = true;
            } finally {
                runBtn.disabled = false;
                isRunning = false;
            }
        });

        // Khởi tạo
        updateFileDisplay(null);
    })();
</script>
</body>
</html>
"""

# ------------------- Endpoints -------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE

@app.get("/upload", response_class=HTMLResponse)
async def upload_form():
    # Chuyển hướng về trang chủ để tránh trùng lặp
    return HTML_PAGE

@app.post("/optimize")
async def optimize(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file Excel (.xlsx / .xls)")

    try:
        contents = await file.read()
        file_like = io.BytesIO(contents)

        # Gọi hàm tối ưu
        excel_buffer, total_rows, total_clashes = run_optimization(file_like)

        # Trả về file với header bổ sung
        response = StreamingResponse(
            io.BytesIO(excel_buffer.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=Ket_qua_phan_bo.xlsx",
                "X-Total-Rows": str(total_rows),
                "X-Total-Clashes": str(total_clashes)
            }
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
