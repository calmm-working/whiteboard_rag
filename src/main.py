import cv2
import uuid
import os
import glob
from dotenv import load_dotenv

# Import các module lõi mà bạn đã viết
from core.yolo_engine import YoloEngine
from core.ocr_engine import TrocrEngine
from core.rag_engine import RAGEngine

# Tự động tìm và nạp các biến môi trường từ file .env
load_dotenv()

# ==========================================
# HÀM XỬ LÝ LUỒNG 1: QUÉT & LƯU TRỮ
# ==========================================
def process_new_image(image_path, yolo_model, ocr_engine, rag_engine):
    print(f"\n--- ĐANG XỬ LÝ ẢNH: {image_path} ---")
    
    # 1. Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Lỗi: Không thể đọc ảnh từ {image_path}")
        return

    # 2. Dùng YoloEngine xịn để bóc tách các dòng chữ
    print("[*] Đang tìm kiếm các khối chữ...")
    # Truyền trực tiếp img (numpy array) vào thay vì đường dẫn
    boxes = yolo_model.get_text_blocks(img, conf_thresh=0.25, iou_thresh=0.4)
    
    if len(boxes) == 0:
        print("⚠️ Không tìm thấy dòng chữ nào trong ảnh!")
        return

    print(f"[*] Tìm thấy {len(boxes)} khối chữ. Đang cho TrOCR đọc từng khối...")
    full_board_text = []

    # 3. Lặp qua các khối chữ đã được YoloEngine sắp xếp chuẩn
    for i, box in enumerate(boxes):
        # YoloEngine trả về trực tiếp tọa độ [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)
        
        # Đảm bảo tọa độ không vượt quá viền ảnh (Tránh lỗi OpenCV)
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop_img = img[y1:y2, x1:x2]
        
        # Chạy AI đọc chữ cho từng khối
        text_line = ocr_engine.predict(crop_img)
        if text_line.strip():
            full_board_text.append(text_line)
            print(f"  + Dòng {i+1}: {text_line}")

    # Gộp tất cả các dòng chữ lại thành một văn bản hoàn chỉnh
    combined_text = "\n".join(full_board_text)
    
    if not combined_text.strip():
        print("⚠️ Ảnh có khung chữ nhưng OCR không thể đọc được nội dung.")
        return

    # 4. Giao cho LangChain tự động Chunking và lưu vào Vector DB
    note_id = f"note_{uuid.uuid4().hex[:8]}"
    metadata = {
        "source_image": os.path.basename(image_path)
    }
    
    rag_engine.add_note_to_db(note_id=note_id, ocr_text=combined_text, metadata=metadata)
    print("-" * 40)

# ==========================================
# HÀM XỬ LÝ LUỒNG 2: HỎI ĐÁP RAG
# ==========================================
def interactive_chat(rag_engine):
    print("\n" + "="*50)
    print("🤖 CHẾ ĐỘ TRỢ LÝ RAG (HỎI ĐÁP VỚI BẢNG TRẮNG)")
    print("="*50)
    print("Nhập 'q' hoặc 'quit' để thoát.\n")
    
    while True:
        question = input("👤 Bạn: ")
        if question.lower() in ['q', 'quit', 'exit']:
            print("👋 Tạm biệt, quay lại menu chính!")
            break
            
        if not question.strip():
            continue
            
        print("⏳ Đang tìm kiếm và tổng hợp câu trả lời...")
        answer = rag_engine.query_with_rerank(question)
        
        print(f"\n🤖 Hệ thống RAG:\n{answer}\n")
        print("-" * 50)

# ==========================================
# HÀM CHÍNH (MAIN ENTRY POINT)
# ==========================================
def main():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HF_REPO_ID = os.getenv("HF_REPO_ID")
    
    if not GEMINI_API_KEY or not HF_REPO_ID:
        print("❌ Lỗi nghiêm trọng: Thiếu GEMINI_API_KEY hoặc HF_REPO_ID trong file .env!")
        return
    
    print("🚀 ĐANG KHỞI ĐỘNG HỆ THỐNG WHITEBOARD RAG...")
    
    try:
        print("[1/3] Đang tải mô hình YOLO...")
        yolo_model = YoloEngine(repo_id=HF_REPO_ID, model_filename="best.pt")
        
        print("[2/3] Đang tải mô hình TrOCR...")
        ocr_engine = TrocrEngine()
        
        print("[3/3] Đang tải LangChain RAG Engine...")
        rag_engine = RAGEngine(api_key=GEMINI_API_KEY)
        
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng khi khởi tạo hệ thống: {e}")
        return
        
    # MENU ĐIỀU HƯỚNG
    while True:
        print("\n" + "="*50)
        print("🎯 BẢNG ĐIỀU KHIỂN CHÍNH")
        print("1. 📸 Tự động quét toàn bộ ảnh trong thư mục './data'")
        print("2. 💬 Hỏi đáp với Trợ lý AI (Truy xuất ghi chú cũ)")
        print("3. ❌ Tắt hệ thống")
        print("="*50)
        
        choice = input("Nhập lựa chọn của bạn (1/2/3): ")
        
        if choice == '1':
            data_folder = "./data"
            
            if not os.path.exists(data_folder):
                print(f"❌ Lỗi: Không tìm thấy thư mục '{data_folder}' trong dự án.")
                continue
                
            # Đã fix lỗi lặp file của Windows bằng set()
            image_files = set()
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                image_files.update(glob.glob(os.path.join(data_folder, ext)))
            
            image_files = list(image_files)
                
            if not image_files:
                print(f"⚠️ Không có bức ảnh nào trong thư mục '{data_folder}'.")
                continue
                
            print(f"\n[*] Tìm thấy {len(image_files)} bức ảnh. Bắt đầu xử lý hàng loạt...")
            
            for img_path in image_files:
                process_new_image(img_path, yolo_model, ocr_engine, rag_engine)
                
            print(f"\n✅ HOÀN TẤT! Đã quét và lưu {len(image_files)} bức ảnh vào Vector Database.")
                
        elif choice == '2':
            interactive_chat(rag_engine)
            
        elif choice == '3':
            print("Đang tắt hệ thống. Hẹn gặp lại! 👋")
            break
            
        else:
            print("⚠️ Lựa chọn không hợp lệ, vui lòng nhập số 1, 2 hoặc 3.")

if __name__ == "__main__":
    main()