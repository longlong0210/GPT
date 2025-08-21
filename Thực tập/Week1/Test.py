from google import genai
import gradio as gr

# Khởi tạo client Gemini
client = genai.Client(api_key="AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs")

# Hàm trả lời câu hỏi
def ask_gemini(question):
    if not question.strip():
        return "⚠️ Vui lòng nhập câu hỏi!"
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# Giao diện Gradio
iface = gr.Interface(
    fn=ask_gemini,
    inputs=gr.Textbox(label="Nhập câu hỏi của bạn", placeholder="Ví dụ: AI là gì?", lines=2),
    outputs=gr.Textbox(label="Trả lời từ Gemini"),
    title="💬 Chatbot Google Gemini",
    description="Nhập câu hỏi vào ô bên dưới và nhận câu trả lời từ Google Gemini."
)

# Chạy giao diện
iface.launch()
