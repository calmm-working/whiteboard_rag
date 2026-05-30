import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    def __init__(self, api_key, db_path="./whiteboard_db"):
        print("[*] Đang khởi tạo RAG Engine với LangChain...")
        
        # Thiết lập biến môi trường cho Gemini
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # 1. Khởi tạo LLM (Gemini 1.5 Flash)
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
        
        # 2. Khởi tạo Embedding (Bi-encoder cho tìm kiếm nhanh)
        self.embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        
        # 3. Khởi tạo Vector Database (Chroma) qua LangChain
        self.vectorstore = Chroma(
            persist_directory=db_path, 
            embedding_function=self.embeddings,
            collection_name="whiteboard_notes"
        )
        
        # 4. Công cụ cắt văn bản thông minh của LangChain
        # Cắt theo ký tự (chia ưu tiên theo đoạn văn \n\n, rồi đến câu \n, rồi khoảng trắng)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Khoảng 1000 ký tự (~200 từ)
            chunk_overlap=200   # Chồng lấp 200 ký tự để giữ ngữ cảnh
        )
        
        # 5. Cấu hình luồng Retrieve & Rerank (Điều kỳ diệu của LangChain)
        # 5.1 - Base Retriever: Lấy Top 10 nhanh bằng Vector
        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # 5.2 - Reranker: Chấm điểm lại thật sâu
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=3) # Lấy Top 3 sau khi chấm điểm
        
        # 5.3 - Gộp lại thành một ống dẫn (Pipeline) duy nhất
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
        print("[+] RAG Engine đã sẵn sàng!")

    def add_note_to_db(self, note_id, ocr_text, metadata=None):
        """
        Lưu văn bản vào DB (LangChain sẽ tự động Chunking)
        """
        # Tạo đối tượng Document chuẩn của LangChain
        meta = metadata or {}
        meta["source_note_id"] = str(note_id)
        doc = Document(page_content=ocr_text, metadata=meta)
        
        # Tự động cắt văn bản
        chunks = self.text_splitter.split_documents([doc])
        
        # Đẩy vào Chroma
        self.vectorstore.add_documents(chunks)
        print(f"[+] Đã băm thành {len(chunks)} khối và lưu ghi chú {note_id} vào DB.")

    def query_with_rerank(self, user_question):
        """
        Luồng hỏi đáp tự động
        """
        # CHỈ 1 DÒNG LỆNH: Langchain tự lo việc Encode -> Tìm Top 10 -> Rerank -> Trả về Top 3
        best_docs = self.compression_retriever.invoke(user_question)
        
        if not best_docs:
            return "Không tìm thấy thông tin nào liên quan trong các bảng trắng đã lưu."
            
        # Tổng hợp nội dung từ các khối tài liệu
        context_text = "\n\n---\n\n".join([doc.page_content for doc in best_docs])
        
        # Tạo Prompt Template chuẩn
        prompt = ChatPromptTemplate.from_template(
            "Dựa vào các trích xuất từ cơ sở dữ liệu ghi chú bảng trắng dưới đây, hãy trả lời câu hỏi của người dùng. "
            "Nếu thông tin không có, hãy nói rõ là không có.\n\n"
            "[GHI CHÚ BẢNG TRẮNG]:\n{context}\n\n"
            "[CÂU HỎI]: {question}"
        )
        
        # Nối Prompt -> LLM -> Bộ lọc dọn rác (StrOutputParser)
        chain = prompt | self.llm | StrOutputParser()
        
        # Chạy đường ống (Bây giờ nó sẽ tự động trả về text sạch)
        answer = chain.invoke({"context": context_text, "question": user_question})
        
        return answer