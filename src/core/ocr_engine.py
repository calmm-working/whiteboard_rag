from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class VietOCREngine:
    def __init__(self, config_name: str = 'vgg_seq2seq', device: str = 'cpu'):
        """Khởi tạo mô hình VietOCR"""
        print(f"⏳ Đang khởi tạo VietOCR ({config_name}) trên {device.upper()}...")
        try:
            config = Cfg.load_config_from_name(config_name)
            config['device'] = device
            self.predictor = Predictor(config)
            print("✅ VietOCR Engine: Sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình VietOCR: {e}")
            raise

    def recognize(self, pil_image) -> str:
        """Nhận diện chữ từ một bức ảnh PIL"""
        try:
            text = self.predictor.predict(pil_image)
            return text
        except Exception as e:
            print(f"Lỗi khi nhận diện chữ: {e}")
            return ""