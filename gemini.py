import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def gemini(input: str):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    messages = [
        SystemMessage("Viết lại câu giữ nguyên ý nghĩa, bảo toàn và xử lý đúng các từ viết tắt. Chỉ xuất câu đã viết lại."),
        HumanMessage(input),
    ]
    result = llm.invoke(messages)
    return result.content
