import re

def clean_text(text: str) -> str:
    """Hàm làm sạch một văn bản thô."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(
        r'[^a-z0-9\sàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳýỵỷỹ.,!?]',
        '',
        text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_multiple_texts(texts: list[str]) -> list[str]:
    """Hàm làm sạch một danh sách văn bản."""
    return [clean_text(text) for text in texts]

def chunk_text_by_sentence(text: str, chunk_size: int = 500) -> list[str]:
    """Phân đoạn văn bản dựa trên câu để đảm bảo tính toàn vẹn ngữ nghĩa."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def chunk_multiple_texts(texts: list[str], chunk_size: int = 500) -> list[list[str]]:
    """Phân đoạn một danh sách văn bản."""
    return [chunk_text_by_sentence(text, chunk_size) for text in texts]

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    
    # =================================================================
    # === PHẦN NHẬP LIỆU TƯƠNG TÁC TỪ NGƯỜI DÙNG ===
    # =================================================================
    raw_texts_list = []
    print("Nhập các đoạn văn bản cần xử lý. Gõ 'xong' hoặc 'done' trên một dòng riêng để kết thúc.")
    
    while True:
        # Lấy đầu vào từ người dùng cho mỗi đoạn
        line = input(f"Đoạn {len(raw_texts_list) + 1}: ")
        
        # Kiểm tra lệnh dừng (không phân biệt chữ hoa/thường)
        if line.lower() in ['xong', 'done']:
            break
        
        # Chỉ thêm vào danh sách nếu người dùng có nhập nội dung
        if line:
            raw_texts_list.append(line)

    # Kiểm tra xem người dùng đã nhập gì chưa
    if not raw_texts_list:
        print("\nBạn chưa nhập đoạn văn bản nào. Chương trình kết thúc.")
    else:
        print("\n" + "="*50)
        print("Bắt đầu xử lý...")
        print("="*50)

        # --- BƯỚC 1: LÀM SẠCH ĐỒNG LOẠT ---
        print("\n--- BƯỚC 1: LÀM SẠCH VĂN BẢN ---")
        cleaned_texts_list = clean_multiple_texts(raw_texts_list)
        print("Đã làm sạch xong.\n")
        print("-" * 50)

        # --- BƯỚC 2: PHÂN ĐOẠN ĐỒNG LOẠT ---
        print("\n--- BƯỚC 2: PHÂN ĐOẠN CÁC VĂN BẢN ĐÃ SẠCH ---")
        chunked_data = chunk_multiple_texts(cleaned_texts_list, chunk_size=100)
        
        for i, chunks in enumerate(chunked_data):
            print(f"\n--- Kết quả phân đoạn cho Đoạn gốc {i+1} ---")
            print(f"Được chia thành {len(chunks)} đoạn nhỏ:")
            for j, chunk in enumerate(chunks):
                print(f"  --- Đoạn con {j+1} (dài {len(chunk)} ký tự) ---")
                print(f"  '{chunk}'")
            
        print("\n" + "-" * 50)
        print("Xử lý hoàn tất!")