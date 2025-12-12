from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

answer_gemini = gemini.invoke("Hello World from Gemini")

print(answer_gemini.content)