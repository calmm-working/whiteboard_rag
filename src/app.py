import gradio as gr
import cv2
import uuid
import os
from dotenv import load_dotenv

# Import các module lõi
from core.yolo_engine import YoloEngine
from core.ocr_engine import TrocrEngine
from core.rag_engine import RAGEngine

# Tải biến môi trường
load_dotenv()

# ==========================================
# 1. KHỞI TẠO HỆ THỐNG GLOBAL
# ==========================================
print("🚀 Đang khởi động lõi AI cho Giao diện...")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_REPO_ID = os.getenv("HF_REPO_ID")

if not GEMINI_API_KEY or not HF_REPO_ID:
    raise ValueError("❌ Lỗi: Thiếu GEMINI_API_KEY hoặc HF_REPO_ID trong file .env!")

# Load các engine 1 lần duy nhất khi bật app
yolo_engine = YoloEngine(repo_id=HF_REPO_ID, model_filename="best.pt")
ocr_engine = TrocrEngine()
rag_engine = RAGEngine(api_key=GEMINI_API_KEY)

# ==========================================
# 2. HÀM XỬ LÝ CHO GIAO DIỆN
# ==========================================
def process_upload(image_path):
    """Xử lý ảnh tải lên từ giao diện và trả về text để hiển thị"""
    if not image_path:
        return "⚠️ Vui lòng tải lên một bức ảnh."
        
    img = cv2.imread(image_path)
    if img is None:
        return "❌ Lỗi: Không thể đọc ảnh."

    boxes = yolo_engine.get_text_blocks(img, conf_thresh=0.1, iou_thresh=0.7)
    if len(boxes) == 0:
        return "⚠️ Không tìm thấy dòng chữ nào trong ảnh!"
    debug_dir = "./data/debug_crops"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"[*] Đang lưu các ảnh cắt vào: {debug_dir}")
    full_board_text = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_img = img[y1:y2, x1:x2]
        debug_filepath = os.path.join(debug_dir, f"crop_{i+1}.jpg")
        cv2.imwrite(debug_filepath, crop_img)
        
        text_line = ocr_engine.predict(crop_img)
        if text_line.strip():
            full_board_text.append(text_line)

    combined_text = "\n".join(full_board_text)
    if not combined_text.strip():
        return "⚠️ Ảnh có khung chữ nhưng OCR không đọc được nội dung."

    # Lưu vào CSDL
    note_id = f"note_{uuid.uuid4().hex[:8]}"
    metadata = {"source_image": os.path.basename(image_path)}
    rag_engine.add_note_to_db(note_id=note_id, ocr_text=combined_text, metadata=metadata)
    
    return f"✅ ĐÃ LƯU THÀNH CÔNG VÀO CƠ SỞ DỮ LIỆU!\n\n📝 Nội dung bóc tách được:\n{'-'*40}\n{combined_text}"

def chat_with_bot(message, history):
    """Hàm hỏi đáp được tối ưu cho giao diện Chat của Gradio"""
    # Hiện tại chúng ta chỉ truyền câu hỏi mới nhất vào RAG (chưa cần lịch sử chat)
    answer = rag_engine.query_with_rerank(message)
    return answer

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN BẰNG GRADIO
# ==========================================
# Đã bỏ tham số theme ở đây theo yêu cầu của Gradio 6.0
with gr.Blocks(title="Whiteboard RAG") as demo:
    gr.Markdown("# 🧠 Hệ thống Đọc hiểu Bảng trắng (Whiteboard RAG)")
    gr.Markdown("Tải ảnh bảng trắng lên để AI bóc tách lưu trữ, sau đó chuyển sang tab Hỏi đáp để truy xuất thông tin.")
    
    # Tạo 2 Tabs (Thẻ) riêng biệt
    with gr.Tabs():
        
        # TAB 1: NHẬP LIỆU
        with gr.TabItem("📸 1. Quét & Lưu Ảnh"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Khung Upload Ảnh
                    img_input = gr.Image(type="filepath", label="Tải ảnh lên (Kéo thả hoặc Click)")
                    upload_btn = gr.Button("🚀 Xử lý & Lưu vào CSDL", variant="primary")
                    
                with gr.Column(scale=1):
                    # Khung hiển thị kết quả
                    upload_output = gr.Textbox(label="Kết quả xử lý", lines=15, interactive=False)
            
            # Gắn sự kiện khi bấm nút
            upload_btn.click(fn=process_upload, inputs=img_input, outputs=upload_output)
            
        # TAB 2: CHATBOT RAG
        with gr.TabItem("💬 2. Trợ lý AI (Hỏi đáp)"):
            # Gradio 5.0+ tự động quản lý UI, không cần truyền các tham số nút bấm cũ
            gr.ChatInterface(
                fn=chat_with_bot,
                examples=["Nội dung chính của các bức ảnh là gì?", "Có nhắc đến thuật toán nào không?"]
            )

# Chạy ứng dụng
if __name__ == "__main__":
    print("🌐 Đang khởi tạo giao diện Web...")
    # Chuyển tham số theme xuống hàm launch()
    demo.launch(inbrowser=True, theme=gr.themes.Soft())