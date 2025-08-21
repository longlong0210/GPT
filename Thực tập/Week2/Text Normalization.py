# Import các thư viện cần thiết
import re
import unicodedata

# DANH SÁCH TỪ DỪNG (STOP WORDS) TIẾNG VIỆT
VIETNAMESE_STOP_WORDS = [
    "và", "là", "mà", "thì", "của", "ở", "tại", "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc",
    "cho", "chứ", "chưa", "có", "có thể", "cùng", "cũng", "đã", "đang", "đây", "để", "đến", "đều", "điều",
    "do", "đó", "được", "gì", "khi", "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "một", "nên", "nếu",
    "ngay", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra", "rằng", "rất", "rồi",
    "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "từng", "vẫn", "vào", "vậy",
    "về", "vì", "việc", "với", "vừa"
]

def normalize_text(text: str) -> str:
    """
    Hàm chuẩn hóa văn bản tiếng Việt.
    """
    # 1. Chuẩn hóa Unicode
    text = unicodedata.normalize('NFC', text)

    # 2. Chuyển về chữ thường
    text = text.lower()

    # 3. Xóa bỏ dấu câu
    text = re.sub(r'[^\w\s]', '', text)

    # 4. Tách từ và loại bỏ từ dừng
    words = text.split()
    filtered_words = [word for word in words if word not in VIETNAMESE_STOP_WORDS]
    text = " ".join(filtered_words)

    # 5. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Người dùng nhập văn bản
    raw_text = input("Nhập đoạn văn bản cần chuẩn hóa:\n")

    print("\n--- VĂN BẢN GỐC ---")
    print(raw_text)
    print("-" * 50)

    # Thực hiện chuẩn hóa
    normalized_text = normalize_text(raw_text)

    print("--- VĂN BẢN ĐÃ ĐƯỢC CHUẨN HÓA ---")
    print(normalized_text)
    print("-" * 50)
