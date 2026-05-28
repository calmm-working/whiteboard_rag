import cv2
import os
from PIL import Image
from core.yolo_engine import YoloEngine
from core.ocr_engine import VietOCREngine
from dotenv import load_dotenv

load_dotenv()
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

def process_whiteboard(image_path: str, yolo: YoloEngine, ocr: VietOCREngine) -> str:
    """Hàm xử lý luồng chính: Nhận ảnh -> Cắt hộp -> Đọc chữ"""
    if not os.path.exists(image_path):
        return f"Lỗi: Không tìm thấy file ảnh tại {image_path}"

    print(f"\n🚀 Đang xử lý ảnh: {image_path}...")
    img = cv2.imread(image_path)
    
    # 1. Tìm các khối chữ bằng YOLO
    boxes = yolo.get_text_blocks(img)
    
    if len(boxes) == 0:
        return "Không tìm thấy chữ nào trên bảng."

    print(f"Khoanh vùng được {len(boxes)} khối chữ. Bắt đầu đọc OCR...\n")
    
    extracted_lines = []
    
    # 2. Cắt từng khối và đưa cho VietOCR đọc
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        crop_img = img[ymin:ymax, xmin:xmax]
        # Thêm 2 dòng này để debug
        debug_filename = f"data/debug_crop_{i}.jpg"
        cv2.imwrite(debug_filename, crop_img)
        
        # Chuyển OpenCV (BGR) sang màu chuẩn (RGB) cho Cấu trúc PIL
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        
        # Đọc chữ
        text = ocr.recognize(pil_img)
        extracted_lines.append(text)
        print(f"  [Dòng {i+1}]: {text}")

    # 3. Gộp tất cả lại thành một văn bản hoàn chỉnh
    final_document = "\n".join(extracted_lines)
    return final_document

if __name__ == "__main__":
    # Khởi tạo 2 Engine
    yolo_engine = YoloEngine(repo_id=HF_REPO_ID)
    ocr_engine = VietOCREngine()
    
    test_image_path = "data/test.jpg" 
    
    print("-" * 50)
    result_text = process_whiteboard(test_image_path, yolo_engine, ocr_engine)
    
    print("-" * 50)
    print("\n📜 KẾT QUẢ VĂN BẢN ĐƯỢC RÚT TRÍCH:\n")
    print(result_text)