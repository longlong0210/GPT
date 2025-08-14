import google.generativeai as genai

# Cấu hình API key
genai.configure(api_key="AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs")

def get_gemini_response(prompt, model_name="gemini-2.5-flash", temperature=0.7):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature
            }
        )
        return response.text.strip()
    except Exception as e:
        return f"Đã xảy ra lỗi khi gọi API Google AI: {str(e)}"


def summarize_text(text_to_summarize):
    prompt = f"Vui lòng tóm tắt đoạn văn bản sau một cách ngắn gọn, chuyên nghiệp và súc tích bằng tiếng Việt:\n\n{text_to_summarize}"
    print("\nĐang tóm tắt...")
    return get_gemini_response(prompt)

def translate_text(text_to_translate, target_language="Tiếng Việt"):
    prompt = f"Dịch đoạn văn bản sau sang {target_language} một cách chính xác và tự nhiên:\n\n{text_to_translate}"
    print(f"\nĐang dịch sang {target_language}...")
    return get_gemini_response(prompt)

def answer_question(context, question):
    prompt = f"Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi được đưa ra. Nếu thông tin không có trong ngữ cảnh, hãy nói 'Tôi không tìm thấy thông tin trong văn bản được cung cấp'.\n\nNgữ cảnh:\n{context}\n\nCâu hỏi: {question}"
    print("\nĐang tìm câu trả lời...")
    return get_gemini_response(prompt, temperature=0.2)

def main():
    while True:
        print("\n--- SCRIPT GOOGLE AI ĐA NĂNG (GEMINI) ---")
        print("1. Tóm tắt văn bản")
        print("2. Dịch văn bản")
        print("3. Trả lời câu hỏi dựa trên văn bản")
        print("4. Thoát")

        choice = input("Nhập lựa chọn của bạn (1/2/3/4): ")

        if choice == '1':
            text = input("Dán văn bản bạn muốn tóm tắt vào đây:\n")
            summary = summarize_text(text)
            print("\n--- BẢN TÓM TẮT ---")
            print(summary)

        elif choice == '2':
            text = input("Dán văn bản bạn muốn dịch vào đây:\n")
            lang = input("Bạn muốn dịch sang ngôn ngữ nào? (Mặc định: Tiếng Việt): ")
            if not lang:
                lang = "Tiếng Việt"
            translation = translate_text(text, lang)
            print(f"\n--- BẢN DỊCH ({lang}) ---")
            print(translation)

        elif choice == '3':
            context = input("Dán văn bản ngữ cảnh vào đây:\n")
            question = input("Nhập câu hỏi của bạn:\n")
            answer = answer_question(context, question)
            print("\n--- CÂU TRẢ LỜI ---")
            print(answer)

        elif choice == '4':
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

if __name__ == "__main__":
    main()
