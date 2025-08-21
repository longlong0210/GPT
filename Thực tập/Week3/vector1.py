# Bước 1: Cài đặt thư viện cần thiết
# Mở terminal hoặc command prompt và chạy lệnh sau:
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Bước 2: Tải một mô hình đã được huấn luyện trước cho Tiếng Việt
# 'bkai-foundation-models/vietnamese-bi-encoder' là một mô hình mạnh mẽ và phổ biến.
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
print("Tải mô hình thành công!")

# Bước 3: Chuẩn bị các câu bạn muốn chuyển đổi thành vector
sentences = [
    "Hà Nội là thủ đô của Việt Nam.",
    "Phở là một món ăn truyền thống nổi tiếng.",
    "Trí tuệ nhân tạo đang thay đổi thế giới.",
    "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất nước."
]

# Bước 4: Mã hóa các câu thành vector
print("\nĐang mã hóa các câu thành vector...")
sentence_embeddings = model.encode(sentences)

# Bước 5: In kết quả để xem
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("\nCâu:", sentence)
    # In 5 giá trị đầu tiên của vector để xem trước
    print("Vector (5 giá trị đầu):", embedding[:5])
    print("Kích thước của vector:", embedding.shape)

# --- VÍ DỤ ỨNG DỤNG: TÌM CÂU TƯƠNG ĐỒNG NHẤT ---
print("\n" + "="*50)
print("VÍ DỤ: TÌM KIẾM NGỮ NGHĨA")
print("="*50)

query = "Thành phố nào là trung tâm kinh tế của Việt Nam?"
query_embedding = model.encode([query])[0]

# Tính độ tương đồng cosine giữa câu truy vấn và các câu trong kho dữ liệu
similarities = cosine_similarity([query_embedding], sentence_embeddings)

# Tìm câu có độ tương đồng cao nhất
most_similar_index = similarities.argmax()
most_similar_sentence = sentences[most_similar_index]
similarity_score = similarities[0][most_similar_index]

print(f"Câu truy vấn: '{query}'")
print(f"Câu tương đồng nhất tìm thấy: '{most_similar_sentence}'")
print(f"Điểm tương đồng: {similarity_score:.4f}")

