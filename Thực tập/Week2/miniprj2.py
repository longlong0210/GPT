import re
import unicodedata
import os
import fitz  # PyMuPDF
import docx

# --- DANH SÁCH TỪ DỪNG ---
VIETNAMESE_STOP_WORDS = [
    "và", "là", "mà", "thì", "của", "ở", "tại", "bị", "bởi", "cả", "các", "cái", "cần",
    "cho", "chứ", "chưa", "có", "cũng", "đã", "đang", "đây", "để", "đến", "đều", "điều",
    "do", "đó", "được", "khi", "không", "lại", "lên", "lúc", "mỗi", "một", "nên", "nếu",
    "ngay", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra", "rằng", "rất",
    "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "vẫn",
    "vào", "vậy", "về", "vì", "việc", "với", "vừa"
]

# --- HÀM ĐỌC FILE ---
def read_document(file_path: str) -> str:
    file_extension = os.path.splitext(file_path)[1].lower()
    content = ""
    try:
        if file_extension == ".pdf":
            with fitz.open(file_path) as doc:
                for page in doc:
                    content += page.get_text()
        elif file_extension == ".docx":
            doc_obj = docx.Document(file_path)
            for para in doc_obj.paragraphs:
                content += para.text + "\n"
        elif file_extension == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            return f"Lỗi: Định dạng file '{file_extension}' không được hỗ trợ."
        return content
    except FileNotFoundError:
        return f"Lỗi: File '{file_path}' không tồn tại."
    except Exception as e:
        return f"Lỗi khi đọc file: {e}"

# --- HÀM LÀM SẠCH ---
def normalize_and_clean_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in VIETNAMESE_STOP_WORDS]
    text = " ".join(filtered_words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- HÀM CHIA ĐOẠN ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    return chunks

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Người dùng nhập đường dẫn file
    document_path = input("Nhập đường dẫn tới file (.pdf, .docx, .txt): ").strip()

    print("--- BẮT ĐẦU QUY TRÌNH CHUẨN HÓA DỮ LIỆU ---")

    raw_content = read_document(document_path)

    if "Lỗi:" in raw_content:
        print(raw_content)
    else:
        print("[1] Đang chuẩn hóa và làm sạch văn bản...")
        normalized_content = normalize_and_clean_text(raw_content)

        print("[2] Đang phân đoạn văn bản...")
        text_chunks = chunk_text(normalized_content, chunk_size=300, chunk_overlap=50)

        print("\n--- KẾT QUẢ ---")
        print(f"Tổng số đoạn: {len(text_chunks)}\n")
        for i, chunk in enumerate(text_chunks):
            print(f"===== ĐOẠN {i+1} =====")
            print(chunk)
            print()

        print("--- KẾT THÚC ---")
