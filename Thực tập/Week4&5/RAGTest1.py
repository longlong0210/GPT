import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import textwrap

# --- PHẦN 1: CẤU HÌNH GOOGLE AI ---
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = "AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs"  # Thay bằng API key của bạn
        print("⚠️ Cảnh báo: API Key đang được đặt trực tiếp trong code.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Đã cấu hình Google AI thành công.")
except Exception as e:
    print(f"❌ Lỗi cấu hình Google AI: {e}")


# --- PHẦN 2: HỆ THỐNG RAG ---
class RAGSystem:
    def __init__(self, corpus, embedding_model_name='bkai-foundation-models/vietnamese-bi-encoder'):
        print("🚀 Đang khởi tạo hệ thống RAG...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash')
        self.vector_db, self.chunks = self._build_vector_db(corpus)
        print("✅ Hệ thống RAG đã sẵn sàng.")

    def _build_vector_db(self, corpus):
        # Chia văn bản thành các đoạn ~200 ký tự
        text_splitter = textwrap.TextWrapper(width=200, break_long_words=False, replace_whitespace=False)
        chunks = text_splitter.wrap(corpus)
        print(f"📄 Tài liệu đã được chia thành {len(chunks)} đoạn.")

        # Tạo embedding
        print("🔄 Đang tạo vector embedding...")
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
        chunk_embeddings = chunk_embeddings.cpu().numpy()

        # FAISS Index
        d = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(chunk_embeddings)
        print("📦 Đã xây dựng xong Vector DB.")
        return index, chunks

    def retrieve(self, query, k=3):
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.vector_db.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]

    def generate(self, relevant_chunks, query):
        context = "\n\n".join(relevant_chunks)
        prompt = f"""
        Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi một cách tự nhiên, chính xác.
        Nếu thông tin không có trong ngữ cảnh, hãy nói rõ điều đó.

        **Ngữ cảnh:**
        {context}

        **Câu hỏi:**
        {query}

        **Câu trả lời:**
        """
        response = self.generation_model.generate_content(prompt)
        return response.text

    def ask(self, query):
        print("\n🔍 Đang truy xuất thông tin...")
        retrieved_chunks = self.retrieve(query)
        print("✍️ Đang tạo câu trả lời...")
        return self.generate(retrieved_chunks, query)


# --- PHẦN 3: CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Nhập dữ liệu và câu hỏi
    corpus_input = input("📥 Nhập dữ liệu văn bản của bạn:\n")
    question_input = input("\n❓ Nhập câu hỏi của bạn:\n")

    # Khởi tạo hệ thống
    rag_system = RAGSystem(corpus_input)

    # Trả lời câu hỏi
    answer = rag_system.ask(question_input)
    print("\n🤖 Câu trả lời từ RAG:")
    print(answer)
