#plan: memory: redis (lam trc)
#      ngat luong ngay trong backend: gui webhook, giam luong model phai dung, human-in-the-loop?
#idea: human-in-the-loop
# hoi lai ad -> send webhook n8n(wflow ngatluong) -> sale send answer(not tinz page) -> webhook n8n(wflow ngatluong)
#-> get sale answer -> send to agent -> answer back to customer -> refine -> embedding -> vector store
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.redis import RedisSaver
from fastapi import FastAPI
from pydantic import BaseModel

llm = ChatOllama(model = "hf.co/MaziyarPanahi/Qwen3-14B-GGUF:Q3_K_L",
                temperature = 0.1,
                top_k = 20,
                top_p = 0.6)

#Prompt
with open('/home/aipencil/DuyNgaDocTon/prompt_template.txt', 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

#State
class State(TypedDict):
    messages: Annotated[list, add_messages]
 
#Node
def chatbot(state: State): #START
    response = llm_with_tools.invoke({"messages": state["messages"]})
    return {"messages": [response]}

from Tools import get_class_info, rag_search, human_assistance
tools = [get_class_info, rag_search, human_assistance]
tool_node = ToolNode(tools) #END (if condition)
llm_with_tools = prompt_template | llm.bind_tools(tools)

#Building graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", 
                                    tools_condition, 
                                    {"tools": "tools", "__end__": "__end__"}
)

#Checkpoint (History)
with RedisSaver.from_conn_string("redis://:12345678Aa@100.107.93.75:6379/0") as checkpointer:
    checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)

class Tinz(BaseModel):
    user: str
    thread_id: str

app = FastAPI()
@app.post("/chat")
async def chat_endpoint(request: Tinz):
    config = {
        "configurable": {
            "thread_id": request.thread_id 
        }
    }

    for event in graph.stream({"messages": [{"role": "user", "content": request.user}]}, config):
        print(event) #for debugging
        for value in event.values():
            assistant = value["messages"][-1].content
            answer = assistant.split("</think>")[-1].replace("\n", " ").strip()
    return answer
# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 
