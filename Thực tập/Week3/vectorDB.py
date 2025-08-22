import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- PHẦN 1: CHUẨN BỊ DỮ LIỆU VÀ EMBEDDING ---

# 1. Chuẩn bị kho dữ liệu văn bản của bạn
documents = [
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam.",
    "Phở là một món ăn truyền thống với nước dùng đậm đà và bánh phở mềm.",
    "Trí tuệ nhân tạo (AI) đang thay đổi nhanh chóng nhiều lĩnh vực của cuộc sống.",
    "Thành phố Hồ Chí Minh, thường được gọi là Sài Gòn, là trung tâm kinh tế lớn nhất Việt Nam.",
    "Bóng đá là môn thể thao vua, được yêu thích trên toàn thế giới.",
    "Học máy là một nhánh quan trọng của trí tuệ nhân tạo.",
    "Bún chả là đặc sản nổi tiếng của ẩm thực Hà Nội."
]

# 2. Tải một mô hình Sentence Transformer để tạo embedding
# 'bkai-foundation-models/vietnamese-bi-encoder' là một lựa chọn mạnh mẽ cho Tiếng Việt.
print("Đang tải mô hình embedding...")
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
print("Tải mô hình thành công!")

# 3. Chuyển đổi toàn bộ kho dữ liệu thành các vector
print("\nĐang tạo vector từ kho dữ liệu...")
document_embeddings = model.encode(documents)

# In ra kích thước để kiểm tra
# (số lượng tài liệu, số chiều của mỗi vector)
print(f"Đã tạo thành công {document_embeddings.shape[0]} vector, mỗi vector có {document_embeddings.shape[1]} chiều.")


# --- PHẦN 2: XÂY DỰNG VÀ LƯU TRỮ VECTOR DATABASE VỚI FAISS ---

# 1. Lấy số chiều của vector
d = document_embeddings.shape[1]

# 2. Xây dựng một chỉ mục (index) cho FAISS
# IndexFlatL2 là loại chỉ mục đơn giản nhất, nó thực hiện tìm kiếm brute-force
# bằng cách so sánh vector truy vấn với tất cả các vector khác.
# Phù hợp cho các bộ dữ liệu nhỏ và vừa.
print("\nĐang xây dựng chỉ mục FAISS...")
index = faiss.IndexFlatL2(d)

# 3. Thêm các vector vào chỉ mục
# FAISS yêu cầu dữ liệu phải ở định dạng float32
index.add(document_embeddings.astype('float32'))

print(f"Đã thêm {index.ntotal} vector vào chỉ mục. Database đã sẵn sàng!")


# --- PHẦN 3: TÌM KIẾM TRONG VECTOR DATABASE ---

print("\n" + "="*50)
print("BẮT ĐẦU TÌM KIẾM NGỮ NGHĨA")
print("="*50)

# 1. Câu truy vấn bạn muốn tìm kiếm
query = "Món ăn nào ngon ở thủ đô?"
k = 3 # Số lượng kết quả gần nhất muốn tìm

# 2. Chuyển câu truy vấn thành vector SỬ DỤNG CÙNG MỘT MÔ HÌNH
query_embedding = model.encode([query])

# 3. Thực hiện tìm kiếm trong chỉ mục
# index.search trả về 2 giá trị:
# D: distances (khoảng cách) - khoảng cách càng nhỏ càng liên quan
# I: indices (chỉ số) - vị trí của các vector kết quả trong mảng gốc
distances, indices = index.search(query_embedding.astype('float32'), k)

# 4. Hiển thị kết quả
print(f"Câu truy vấn: '{query}'")
print(f"Đã tìm thấy {k} kết quả liên quan nhất:\n")

for i, idx in enumerate(indices[0]):
    print(f"--- Kết quả {i+1} ---")
    print(f"Câu gốc: {documents[idx]}")
    print(f"Khoảng cách (L2 distance): {distances[0][i]:.4f}")
    print()

