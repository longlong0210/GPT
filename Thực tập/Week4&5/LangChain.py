import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import RetrievalQA

# ====== 1. Cấu hình API key ======
os.environ["GOOGLE_API_KEY"] = "AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs"  # <-- thay bằng key thật

# ====== 2. Dữ liệu mẫu ======
raw_text = """
Tấn công brute-force là một phương pháp bẻ khóa phổ biến . Một cuộc tấn công brute-force liên quan đến việc 'đoán' tên người dùng và mật khẩu để truy cập trái phép vào hệ thống, hacker sẽ sử dụng phương pháp thử và sai để cố gắng đoán thông tin đăng nhập hợp lệ. Ngoài ra ta có thể sử dụng brute-force để khai thác OTP, Timestamp , Cookie, vv... Tuy nhiên bài viết này sẽ tập trung vào brute-force mật khẩu để đăng nhập tài khoản người dùng.

Các cuộc tấn công này thường được tự động hóa bằng cách sử dụng danh sách các từ gồm tên người dùng và mật khẩu thường được sử dụng để có thể đạt được kết quả tốt nhất. Việc sử dụng các công cụ chuyên dụng có khả năng cho phép hacker thực hiện đăng nhập 1 cách tự động nhiều lần với tốc độ cao.

Brute-force là một phương pháp tấn công đơn giản và có tỷ lệ thành công cao. Bởi vì tùy thuộc vào độ dài và độ phức tạp của mật khẩu, việc bẻ khóa mật khẩu có thể mất từ vài giây đến nhiều năm. Do đó các trang web sử dụng phương thức đăng nhập dựa trên mật khẩu hoàn toàn có thể rất dễ bị tấn công nếu họ không thực hiện đầy đủ biện pháp bảo vệ bạo lực.
""".strip()

# ====== 3. Tách văn bản theo câu & chunk “đẹp” ======
# Ưu tiên ngắt theo đoạn, xuống dòng, dấu câu để tránh chẻ ngang câu
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "? ", "! ", "; "],
    keep_separator=False,
    chunk_size=500,       # tăng size để giữ nguyên câu/đoạn
    chunk_overlap=80      # chồng lấp nhẹ để không mất ngữ cảnh
)

base_doc = Document(page_content=raw_text, metadata={"source": "kb"})
chunks = splitter.split_documents([base_doc])

# Gắn thêm metadata nhận diện chunk
for i, d in enumerate(chunks):
    d.metadata.update({"chunk_id": i})

# ====== 4. Embedding + FAISS ======
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = FAISS.from_documents(chunks, embeddings)

# ====== 5. LLM (Gemini) ======
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# ====== 6. Retriever (MMR để giảm trùng lặp) ======
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,          # số chunk cuối cùng trả về
        "fetch_k": 15,   # số lượng lấy rộng ban đầu trước khi MMR chọn lọc
        "lambda_mult": 0.5
    }
)

# ====== 7. Chain RAG ======
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ====== 8. Truy vấn thử ======
query = "Brute force là gì và tấn công như thế nào?"
result = qa_chain.invoke({"query": query})

print("💬 Câu hỏi:", query)
print("🤖 Trả lời:", result["result"])

print("\n📚 Nguồn dữ liệu (chunk đầy đủ):")
for doc in result["source_documents"]:
    cid = doc.metadata.get("chunk_id", "?")
    print(f"\n--- Chunk #{cid} ---")
    print(doc.page_content.strip())
