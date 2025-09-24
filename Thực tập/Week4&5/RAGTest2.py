import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import textwrap
import gradio as gr

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
        text_splitter = textwrap.TextWrapper(width=200, break_long_words=False, replace_whitespace=False)
        chunks = text_splitter.wrap(corpus)
        print(f"📄 Tài liệu đã được chia thành {len(chunks)} đoạn.")

        print("🔄 Đang tạo vector embedding...")
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        chunk_embeddings = chunk_embeddings.cpu().numpy()

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
        retrieved_chunks = self.retrieve(query)
        return self.generate(retrieved_chunks, query)


# --- PHẦN 3: HÀM XỬ LÝ GRADIO ---
def rag_answer(corpus, question):
    if not corpus.strip():
        return "⚠️ Vui lòng nhập văn bản corpus."
    if not question.strip():
        return "⚠️ Vui lòng nhập câu hỏi."

    rag_system = RAGSystem(corpus)
    return rag_system.ask(question)


# --- PHẦN 4: GIAO DIỆN GRADIO ---
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 RAG QA với Google Gemini + FAISS + SentenceTransformer")
    corpus_input = gr.Textbox(label="📄 Văn bản Corpus", placeholder="Nhập nội dung văn bản của bạn", lines=8)
    question_input = gr.Textbox(label="❓ Câu hỏi", placeholder="Nhập câu hỏi", lines=2)
    output_box = gr.Textbox(label="🤖 Câu trả lời", lines=6)

    submit_btn = gr.Button("🚀 Truy vấn RAG")
    submit_btn.click(fn=rag_answer, inputs=[corpus_input, question_input], outputs=output_box)

demo.launch()
