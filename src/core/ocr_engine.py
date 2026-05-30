import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np

class TrocrEngine:
    def __init__(self, model_id="calmm-m/trocr_whiteboard"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Đang tải mô hình TrOCR từ Hugging Face: {model_id}...")
        print(f"[*] Thiết bị đang sử dụng: {self.device.type.upper()}")
        
        # 🟢 ĐÃ SỬA DÒNG NÀY: Mượn Tokenizer (Từ điển) trực tiếp từ Microsoft
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        
        # Não bộ (Trọng số đã fine-tune) thì vẫn lấy từ kho của bạn!
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
        
        print("[+] Khởi tạo TrOCR thành công!")

    def predict(self, image):
        """
        Nhận vào ảnh cắt (numpy array từ OpenCV hoặc PIL Image) và trả về text
        """
        if isinstance(image, np.ndarray):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError("Định dạng ảnh không được hỗ trợ!")

        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()