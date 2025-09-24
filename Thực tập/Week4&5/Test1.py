import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# --- PHẦN 1: CẤU HÌNH VÀ KHỞI TẠO CÁC THÀNH PHẦN CỐT LÕI ---

# Cấu hình API Key của Google AI
# Để an toàn, hãy đặt API key của bạn vào biến môi trường.
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        # Thay thế bằng API key của bạn ở đây nếu không đặt biến môi trường
        GOOGLE_API_KEY = "AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs"
        print("Cảnh báo: API Key được đặt trực tiếp trong code.")
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    print("Đã cấu hình Google AI API Key thành công.")
except Exception as e:
    print(f"Lỗi cấu hình API Key: {e}")

# Khởi tạo các mô hình (tải một lần để tiết kiệm thời gian)
print("Đang tải các mô hình AI...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
print("Tải mô hình thành công!")

# --- PHẦN 2: CÁC HÀM XỬ LÝ CHO GRADIO ---

def create_chatbot(knowledge_base_text):
    """
    Xây dựng chatbot từ văn bản kiến thức do người dùng cung cấp.
    """
    if not knowledge_base_text or knowledge_base_text.strip() == "":
        return None, "Vui lòng nhập cơ sở kiến thức trước khi xây dựng."

    # Chuyển văn bản thô thành đối tượng Document của LangChain
    documents = [Document(page_content=knowledge_base_text)]

    # Phân đoạn văn bản
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Tạo embedding và Vector DB
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Tạo retriever
    retriever = vector_db.as_retriever()

    # --- BẮT ĐẦU PHẦN THAY ĐỔI ---
    # Tạo prompt để biến câu hỏi nối tiếp thành câu hỏi độc lập
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Tạo prompt để trả lời câu hỏi cuối cùng dựa trên ngữ cảnh
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Tạo chuỗi xử lý tài liệu
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Kết hợp tất cả lại thành chuỗi RAG cuối cùng
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain, "Chatbot đã sẵn sàng! Bạn có thể bắt đầu đặt câu hỏi."

def handle_user_query(user_input, chat_history, chatbot_chain):
    """
    Xử lý câu hỏi của người dùng và trả về câu trả lời.
    """
    if chatbot_chain is None:
        return "", chat_history + [[user_input, "Lỗi: Vui lòng xây dựng chatbot trước."]]

    # Chuyển đổi lịch sử chat của Gradio sang định dạng của LangChain
    chat_history_for_chain = []
    for human, ai in chat_history:
        chat_history_for_chain.append(HumanMessage(content=human))
        chat_history_for_chain.append(ai) # Gradio lưu AIMessage dưới dạng str

    # Lấy câu trả lời từ chuỗi RAG
    response = chatbot_chain.invoke({
        "input": user_input,
        "chat_history": chat_history_for_chain
    })
    
    # Cập nhật lịch sử chat của Gradio
    chat_history.append([user_input, response['answer']])
    
    return "", chat_history
# --- KẾT THÚC PHẦN THAY ĐỔI ---

# --- PHẦN 3: XÂY DỰNG GIAO DIỆN GRADIO ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Lưu trữ đối tượng chatbot chain giữa các lần gọi
    chatbot_state = gr.State()

    gr.Markdown("# 🤖 Chatbot Hỏi-Đáp Thông Minh (RAG)")
    gr.Markdown("Nhập tài liệu của bạn vào ô bên dưới, nhấn 'Xây dựng Chatbot', sau đó bắt đầu cuộc trò chuyện.")

    with gr.Row():
        with gr.Column(scale=1):
            knowledge_box = gr.Textbox(
                lines=20,
                label="Cơ sở kiến thức",
                placeholder="Dán nội dung văn bản của bạn vào đây..."
            )
            build_button = gr.Button("Xây dựng Chatbot", variant="primary")
            status_display = gr.Markdown("")

        with gr.Column(scale=2):
            chatbot_display = gr.Chatbot(label="Hộp thoại Chat", height=500)
            query_box = gr.Textbox(
                label="Nhập câu hỏi của bạn",
                placeholder="Ví dụ: Thủ đô của Pháp là gì?"
            )
            submit_button = gr.Button("Gửi")

    # Xử lý sự kiện
    build_button.click(
        fn=create_chatbot,
        inputs=[knowledge_box],
        outputs=[chatbot_state, status_display]
    )

    query_box.submit(
        fn=handle_user_query,
        inputs=[query_box, chatbot_display, chatbot_state],
        outputs=[query_box, chatbot_display]
    )
    
    submit_button.click(
        fn=handle_user_query,
        inputs=[query_box, chatbot_display, chatbot_state],
        outputs=[query_box, chatbot_display]
    )

# --- PHẦN 4: CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    demo.launch(debug=True)
