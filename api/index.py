# api/index.py
import io
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import sys
import os

# Thêm đường dẫn để import ALLOCATION.py từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ALLOCATION import run_optimization

app = FastAPI(title="Container Allocation Optimizer")

@app.get("/")
async def root():
    return {"message": "Upload an Excel file to /optimize"}

@app.post("/optimize")
async def optimize(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are accepted")
    
    try:
        # Đọc nội dung file upload vào BytesIO
        contents = await file.read()
        file_like = io.BytesIO(contents)

        # Gọi hàm tối ưu (trả về buffer, số dòng, số clash)
        excel_buffer, total_rows, total_clashes = run_optimization(file_like)
        
        # Tạo response trả về file
        response = StreamingResponse(
            io.BytesIO(excel_buffer.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=optimized_{file.filename}"
            }
        )
        return response
    except Exception as e:
        # Bắt lỗi và trả về 500 với thông báo
        raise HTTPException(status_code=500, detail=str(e))