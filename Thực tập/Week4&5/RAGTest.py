import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import textwrap

# --- PHẦN 1: CẤU HÌNH VÀ CHUẨN BỊ ---

# Cấu hình API Key của Google AI
# Để an toàn, hãy đặt API key của bạn vào biến môi trường
# hoặc dùng tính năng "Secrets" của Google Colab.
try:
    # from google.colab import userdata
    # GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        # Thay thế bằng API key của bạn ở đây nếu không đặt biến môi trường
        GOOGLE_API_KEY = "AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs"
        print("Cảnh báo: API Key được đặt trực tiếp trong code.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Đã cấu hình Google AI thành công.")
except Exception as e:
    print(f"Lỗi cấu hình Google AI: {e}")


# Kho kiến thức (Knowledge Base) của chúng ta
knowledge_base_text = """
Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính, tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người. Các ứng dụng chính của AI bao gồm học máy, xử lý ngôn ngữ tự nhiên, thị giác máy tính và robot học.

Trong lĩnh vực y tế, AI đang tạo ra một cuộc cách mạng. Các thuật toán có thể phân tích hình ảnh y tế như X-quang và MRI để phát hiện sớm các dấu hiệu của bệnh ung thư với độ chính xác cao. AI cũng giúp các nhà nghiên cứu phân tích một lượng lớn dữ liệu di truyền để phát triển các loại thuốc mới và các phương pháp điều trị cá nhân hóa.

Xe tự lái là một trong những thành tựu nổi bật nhất của AI. Chúng sử dụng một loạt các cảm biến, camera và thuật toán phức tạp để nhận dạng môi trường xung quanh, bao gồm người đi bộ, các phương tiện khác và biển báo giao thông, từ đó đưa ra quyết định lái xe an toàn.

Đối với ngành tài chính, AI được sử dụng để phát hiện các giao dịch gian lận. Các mô hình học máy có thể phân tích hàng triệu giao dịch trong thời gian thực để xác định các mẫu bất thường có thể chỉ ra hành vi lừa đảo. Ngoài ra, các chatbot do AI cung cấp dịch vụ hỗ trợ khách hàng 24/7.

Mặc dù có nhiều tiềm năng, sự phát triển của AI cũng đặt ra các thách thức về đạo đức. Một trong những vấn đề lớn nhất là sự thiên vị trong thuật toán (algorithmic bias), có thể dẫn đến các quyết định không công bằng. Việc đảm bảo AI được phát triển và sử dụng một cách có trách nhiệm là ưu tiên hàng đầu.
"""

# --- PHẦN 2: XÂY DỰNG HỆ THỐNG RAG ---

class RAGSystem:
    def __init__(self, corpus, embedding_model_name='bkai-foundation-models/vietnamese-bi-encoder'):
        """
        Khởi tạo hệ thống RAG.
        """
        print("Đang khởi tạo hệ thống RAG...")
        # Tải mô hình embedding
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Tải mô hình tạo sinh (Gemini)
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Xây dựng Vector DB
        self.vector_db, self.chunks = self._build_vector_db(corpus)
        print("Hệ thống RAG đã sẵn sàng.")

    def _build_vector_db(self, corpus):
        """
        Xây dựng cơ sở dữ liệu vector từ văn bản gốc.
        """
        # Phân đoạn văn bản
        text_splitter = textwrap.TextWrapper(width=600, break_long_words=False, replace_whitespace=False)
        chunks = text_splitter.wrap(corpus)
        print(f"Tài liệu đã được chia thành {len(chunks)} đoạn.")

        # Tạo embedding cho các đoạn
        print("Đang tạo vector embedding...")
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
        chunk_embeddings = chunk_embeddings.cpu().numpy()

        # Xây dựng chỉ mục FAISS
        d = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(chunk_embeddings)
        print("Đã xây dựng xong Vector DB.")
        return index, chunks

    def retrieve(self, query, k=3):
        """
        Truy xuất các đoạn văn bản liên quan từ Vector DB.
        """
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.vector_db.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]

    def generate(self, relevant_chunks, query):
        """
        Tạo câu trả lời từ các đoạn văn bản đã truy xuất.
        """
        # Tạo prompt tăng cường
        context = "\n\n".join(relevant_chunks)
        prompt = f"""
        Dựa vào ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng một cách tự nhiên và chính xác.
        Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không tìm thấy thông tin.

        **Ngữ cảnh:**
        {context}

        **Câu hỏi:**
        {query}

        **Câu trả lời:**
        """
        
        # Gọi Gemini để tạo câu trả lời
        response = self.generation_model.generate_content(prompt)
        return response.text

    def ask(self, query):
        """
        Thực hiện toàn bộ quy trình RAG: Truy xuất và Tạo sinh.
        """
        print("\n" + "="*50)
        print(f"Đang xử lý câu hỏi: '{query}'")
        
        # 1. Retrieval
        print("Bước 1: Đang truy xuất thông tin liên quan...")
        retrieved_chunks = self.retrieve(query)
        
        # 2. Generation
        print("Bước 2: Đang tạo câu trả lời từ thông tin đã truy xuất...")
        final_answer = self.generate(retrieved_chunks, query)
        
        print("="*50)
        return final_answer

# --- PHẦN 3: CHẠY ỨNG DỤNG ---

if __name__ == "__main__":
    # Khởi tạo hệ thống RAG với kho kiến thức
    rag_system = RAGSystem(knowledge_base_text)

    # Đặt câu hỏi cho hệ thống
    question_1 = "AI giúp ích gì cho việc phát hiện gian lận trong tài chính?"
    answer_1 = rag_system.ask(question_1)
    
    print(f"\n❓ Câu hỏi: {question_1}")
    print(f"🤖 Câu trả lời từ RAG:\n{answer_1}")

    # Đặt một câu hỏi khác
    question_2 = "Làm thế nào xe tự lái nhận biết được môi trường xung quanh?"
    answer_2 = rag_system.ask(question_2)

    print(f"\n❓ Câu hỏi: {question_2}")
    print(f"🤖 Câu trả lời từ RAG:\n{answer_2}")
