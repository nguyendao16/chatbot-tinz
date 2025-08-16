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
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.redis import RedisSaver
from fastapi import FastAPI
from pydantic import BaseModel
import json

llm = ChatOllama(model = "qwen3:14b",
                temperature = 0.1,
                top_k = 20,
                top_p = 0.6,
                num_gpu_layers = 999)

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
    mid: str
 
#Node
def chatbot(state: State): #START
    response = llm_with_tools.invoke({"messages": state["messages"]})
    return {"messages": [response]}

from Tools import get_class_info, rag_search, human_assistance
from langchain_core.messages import ToolMessage
tools = [get_class_info, rag_search, human_assistance]
class ToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        mid = inputs.get("mid")
        for tool_call in message.tool_calls:
            #Muc dich la de lay dua message id vao ben trong parameter, de khi can, co the lay duoc username
            if tool_call["name"] == "human_assistance":
                tool_call["args"]["mid"] = mid
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
            else:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False, indent=2),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


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
    mid: str

from gemini import gemini
app = FastAPI()
@app.post("/chat")
async def chat_endpoint(request: Tinz):
    config = {
        "configurable": {
            "thread_id": request.thread_id 
        }
    }
    #refine user message
    user_message = gemini(request.user)

    graph_input = {"messages": [{"role": "user", "content": user_message}], "mid": request.mid}
    for event in graph.stream(graph_input, config):
        #print(event) #for debugging
        for value in event.values():
            assistant = value["messages"][-1].content
            answer = assistant.split("</think>")[-1].replace("\n", " ").strip()
    #print(graph.get_state(config))
    return answer
    
# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 
