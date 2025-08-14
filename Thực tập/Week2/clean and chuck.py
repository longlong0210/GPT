import re

def clean_text(text: str) -> str:
    """
    Hàm làm sạch văn bản thô.
    """
    # 1. Chuyển thành chữ thường
    text = text.lower()

    # 2. Loại bỏ URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Loại bỏ thẻ HTML
    text = re.sub(r'<.*?>', '', text)

    # 4. Loại bỏ các ký tự không mong muốn, giữ tiếng Việt & dấu câu cơ bản
    text = re.sub(
        r'[^a-z0-9\sàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳýỵỷỹ.,!?]',
        '',
        text
    )

    # 5. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def chunk_text(text: str, chunk_size: int = 100, chunk_overlap: int = 50) -> list[str]:
    """
    Phân đoạn văn bản thành các mẩu nhỏ hơn với sự chồng lấn.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += chunk_size - chunk_overlap
    
    return chunks

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Nhập văn bản từ người dùng
    raw_text = input("Nhập đoạn văn bản cần xử lý:\n")

    # 1. Làm sạch văn bản
    print("\n--- BƯỚC 1: LÀM SẠCH VĂN BẢN ---")
    cleaned_text = clean_text(raw_text)
    print("\nVăn bản đã làm sạch:\n", cleaned_text)
    print("-" * 50)

    # 2. Phân đoạn văn bản
    print("\n--- BƯỚC 2: PHÂN ĐOẠN VĂN BẢN ---")
    # Dùng chunk_size nhỏ để dễ kiểm tra kết quả
    text_chunks = chunk_text(cleaned_text, chunk_size=50, chunk_overlap=25)
    print(f"\nVăn bản được chia thành {len(text_chunks)} đoạn:\n")

    for i, chunk in enumerate(text_chunks):
        print(f"--- Đoạn {i+1} (dài {len(chunk)} ký tự) ---")
        print(chunk)
        print()
