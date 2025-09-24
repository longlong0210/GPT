import os
import gradio as gr
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.docstore.document import Document

# Cấu hình API key Google
os.environ["GOOGLE_API_KEY"] = "AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs"  # Thay bằng API key của bạn
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Tạo embeddings & LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# Biến lưu Vector DB (toàn cục)
vectorstore = None

def save_text_and_answer(text_input, question_input):
    global vectorstore
    
    # Nếu nhập văn bản mới → Tạo lại vectorstore
    if text_input:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text_input)
        docs = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Nếu chưa có dữ liệu thì báo lỗi
    if vectorstore is None:
        return "⚠️ Vui lòng nhập văn bản trước khi đặt câu hỏi."
    
    # Tạo chain QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Trả lời câu hỏi
    result = qa_chain.invoke({"query": question_input})
    answer = result["result"]

    # Nguồn tham khảo
    sources = "\n".join([doc.page_content for doc in result["source_documents"]])
    return f"**Câu trả lời:**\n{answer}\n\n**Nguồn tham khảo:**\n{sources}"

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 📚 Chatbot Q&A với Google AI + Vector DB")
    text_input = gr.Textbox(lines=5, placeholder="Nhập văn bản ở đây...", label="Văn bản")
    question_input = gr.Textbox(lines=2, placeholder="Nhập câu hỏi...", label="Câu hỏi")
    output = gr.Markdown()
    submit_btn = gr.Button("Gửi câu hỏi")
    
    submit_btn.click(save_text_and_answer, inputs=[text_input, question_input], outputs=output)

demo.launch()
