import google.generativeai as genai

class LLMEngine:
    def __init__(self, api_key):
        # Khởi tạo kết nối với Gemini API
        genai.configure(api_key=api_key)
        
        # Sử dụng bản flash để tối ưu tốc độ phản hồi
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        print("[+] Khởi tạo LLM Engine thành công!")

    def format_and_correct_notes(self, raw_ocr_text):
        """
        Nhiệm vụ 1: Chữa lỗi chính tả và cấu trúc lại nội dung bảng trắng
        """
        prompt = f"""
        Dưới đây là đoạn văn bản được trích xuất từ bảng trắng thông qua công cụ nhận diện chữ viết tay (OCR). 
        Do hạn chế của công nghệ, văn bản có thể bị sai chính tả, lộn xộn, hoặc thiếu logic.

        Nhiệm vụ của bạn:
        1. Sửa các lỗi chính tả cho tự nhiên và đúng ngữ cảnh.
        2. Cấu trúc lại nội dung cho dễ đọc (dùng tiêu đề, gạch đầu dòng nếu cần).
        3. Tuyệt đối không tự bịa thêm thông tin không có trong văn bản gốc.

        Văn bản OCR thô:
        ---
        {raw_ocr_text}
        ---
        """
        response = self.model.generate_content(prompt)
        return response.text

    def chat_with_notes(self, context_text, user_question):
        """
        Nhiệm vụ 2: Trả lời câu hỏi của người dùng dựa trên nội dung bảng (Phần Generation trong RAG)
        """
        prompt = f"""
        Dựa vào nội dung ghi chú trên bảng trắng dưới đây, hãy trả lời câu hỏi của người dùng.
        Nếu thông tin không có trong ghi chú, hãy nói rõ là không có, đừng tự bịa ra.

        Nội dung bảng trắng:
        ---
        {context_text}
        ---

        Câu hỏi của người dùng: {user_question}
        """
        response = self.model.generate_content(prompt)
        return response.text