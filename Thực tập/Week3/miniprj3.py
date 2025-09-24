# Bước 1: Cài đặt các thư viện cần thiết
# Mở terminal và chạy lệnh sau:
# pip install sentence-transformers faiss-cpu numpy

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

# --- HÀM CHIA VĂN BẢN THEO 2-3 CÂU ---
def split_text_by_sentences(text, sentences_per_chunk=3):
    """
    Chia văn bản thành các đoạn, mỗi đoạn gồm n câu (mặc định 3).
    """
    # Tách câu dựa trên dấu kết thúc câu (.!?), giữ nguyên dấu câu
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks

# --- PHẦN 1: CHUẨN BỊ KIẾN THỨC (KNOWLEDGE BASE) ---
knowledge_base_text = """
Trí tuệ nhân tạo, hay AI, là một trong những lĩnh vực công nghệ phát triển nhanh nhất và có ảnh hưởng sâu rộng nhất trong thế kỷ 21. Về cơ bản, AI là ngành khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người, chẳng hạn như học hỏi, suy luận, giải quyết vấn đề, nhận dạng giọng nói và ra quyết định.

Một trong những ứng dụng đột phá nhất của AI là trong lĩnh vực y tế. Các thuật toán học máy có thể phân tích hình ảnh y tế như X-quang, MRI với độ chính xác cao hơn con người, giúp phát hiện sớm các bệnh nguy hiểm như ung thư. AI cũng được sử dụng để phân tích dữ liệu di truyền, tìm ra các phương pháp điều trị cá nhân hóa cho từng bệnh nhân.

Trong ngành công nghiệp ô tô, xe tự lái là một ví dụ điển hình về sức mạnh của AI. Các hệ thống này sử dụng camera, cảm biến và thuật toán phức tạp để nhận dạng môi trường xung quanh, từ đó đưa ra các quyết định lái xe an toàn. Điều này hứa hẹn sẽ giảm thiểu tai nạn giao thông do lỗi của con người.

Lĩnh vực tài chính cũng được hưởng lợi rất nhiều. Các ngân hàng sử dụng AI để phát hiện giao dịch gian lận trong thời gian thực, phân tích rủi ro tín dụng và tự động hóa các quy trình dịch vụ khách hàng thông qua chatbot thông minh.

Giáo dục cũng không nằm ngoài xu hướng. Các nền tảng học tập cá nhân hóa sử dụng AI để theo dõi tiến độ của học sinh và đề xuất các bài tập, tài liệu phù hợp với năng lực của từng người, giúp tối ưu hóa quá trình học tập.

Tuy nhiên, sự phát triển của AI cũng đặt ra nhiều thách thức, đặc biệt là các vấn đề về đạo đức như sự thiên vị trong thuật toán (algorithmic bias) và vấn đề việc làm khi tự động hóa thay thế lao động con người. Việc xây dựng một khung pháp lý và đạo đức vững chắc là điều cần thiết để đảm bảo AI phát triển một cách có trách nhiệm.
"""

# --- PHẦN 2: XÂY DỰNG VECTOR DATABASE ---
def build_vector_database(text_corpus, model):
    """
    Nhận vào kho văn bản và mô hình embedding, trả về chỉ mục FAISS.
    """
    # Chia văn bản thành các đoạn gồm 2–3 câu
    chunks = split_text_by_sentences(text_corpus, sentences_per_chunk=3)
    print(f"Đã chia tài liệu thành {len(chunks)} đoạn.")

    # Tạo embedding cho từng đoạn
    print("Đang tạo vector cho từng đoạn...")
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    chunk_embeddings = chunk_embeddings.cpu().numpy()

    # Xây dựng chỉ mục FAISS
    print("Đang xây dựng chỉ mục Vector DB (FAISS)...")
    d = chunk_embeddings.shape[1]  # Số chiều của vector
    index = faiss.IndexFlatL2(d)
    index.add(chunk_embeddings)

    print(f"Vector DB đã sẵn sàng với {index.ntotal} vector.")
    return index, chunks

# --- PHẦN 3: HÀM TRUY VẤN ---
def search_relevant_chunks(query, model, index, chunks, k=3):
    """
    Tìm kiếm các đoạn văn bản liên quan nhất đến câu truy vấn.
    """
    print("\n" + "="*50)
    print(f"Đang thực hiện truy vấn: '{query}'")

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# --- PHẦN 4: CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    print("Đang tải mô hình Sentence Transformer...")
    embedding_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    print("Tải mô hình thành công!")

    vector_db_index, original_chunks = build_vector_database(knowledge_base_text, embedding_model)

    user_query = "AI có thể giúp gì trong ngành y tế?"
    results = search_relevant_chunks(user_query, embedding_model, vector_db_index, original_chunks)

    print("\n--- CÁC ĐOẠN VĂN BẢN LIÊN QUAN NHẤT ĐÃ TÌM THẤY ---")
    for i, chunk in enumerate(results):
        print(f"\n--- Kết quả {i+1} ---")
        print(chunk)
