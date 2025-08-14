from google import genai

client = genai.Client(api_key="AIzaSyABLonsDEQ7veJFWZf6lLlHvtPw9K4lBMs")

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Kể tên 3 con vật"
)
print(response.text)