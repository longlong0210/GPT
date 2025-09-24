import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import re

# 1. Hàm đọc file txt hoặc docx từ đường dẫn
def read_file(file_path):
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    return text

# 2. Hàm tách câu và chia thành chunk 10 câu
def split_into_chunks(text, sentences_per_chunk=10):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỹ])', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks

# 3. Tải model tiếng Việt
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

# 4. Hàm tìm kiếm chunk liên quan nhất
def search_best_sentences(data_chunks, query, top_n=3):
    # Tìm chunk liên quan nhất
    chunk_embeddings = model.encode(data_chunks)
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_chunk_index = similarities.argmax()
    best_chunk = data_chunks[top_chunk_index]

    # Tách câu trong chunk đó
    sentences = re.split(r'(?<=[.!?])\s+', best_chunk.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # Tính similarity từng câu
    sentence_embeddings = model.encode(sentences)
    sentence_scores = cosine_similarity([query_embedding], sentence_embeddings)[0]

    # Lấy top_n câu tốt nhất
    top_indices = sentence_scores.argsort()[-top_n:][::-1]
    best_sentences = [f"{sentences[i]} (score: {sentence_scores[i]:.4f})" for i in top_indices]

    return "\n".join(best_sentences)

# 5. Hàm xử lý toàn bộ
def process_data(file_path, text_input, query):
    if file_path is not None:
        text = read_file(file_path)
    else:
        text = text_input
    
    if not text.strip():
        return "Kho dữ liệu trống!"

    chunks = split_into_chunks(text, sentences_per_chunk=10)
    result = search_best_sentences(chunks, query, top_n=3)
    
    return f"🔍 Các câu liên quan nhất:\n{result}"

# 6. Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Tìm kiếm câu liên quan nhất (txt & docx)")
    
    with gr.Row():
        file_input = gr.File(label="Upload file (.txt hoặc .docx)", type="filepath")
        text_input = gr.Textbox(label="Hoặc nhập trực tiếp kho dữ liệu", lines=6)
    
    query_input = gr.Textbox(label="Nhập câu hỏi", lines=2)
    search_btn = gr.Button("Tìm kiếm")
    output_box = gr.Textbox(label="Kết quả", lines=10)
    
    search_btn.click(fn=process_data, 
                     inputs=[file_input, text_input, query_input],
                     outputs=output_box)

if __name__ == "__main__":
    demo.launch()
