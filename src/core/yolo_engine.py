import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

class YoloEngine:
    def __init__(self, repo_id: str, model_filename: str = "best.pt"):
        """Khởi tạo và tải mô hình YOLO từ Hugging Face"""
        print(f"⏳ Đang khởi tạo YOLO từ {repo_id}...")
        try:
            # Tự động tải từ Hugging Face (sẽ dùng cache nếu đã tải rồi)
            self.model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
            self.model = YOLO(self.model_path)
            print("✅ YOLO Engine: Sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình YOLO: {e}")
            raise

    def get_text_blocks(self, img, conf_thresh=0.25, iou_thresh=0.4):
        """
        Quét ảnh và trả về danh sách tọa độ các hộp chữ đã được sắp xếp.
        """
        results = self.model(img, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            return []

        # Sắp xếp tọa độ: Ưu tiên từ trên xuống dưới, sau đó từ trái sang phải
        sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return sorted_boxes