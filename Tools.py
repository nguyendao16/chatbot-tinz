from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
import psycopg2
import json
from langgraph.types import Command, interrupt
embeddings = OllamaEmbeddings(model = "bge-m3:latest")

@tool
def get_class_info(mos = None, Class = None):
    """
    Retrieve class name, language and mos version about the available classes base on MOS version or Class name.
    This tool does not contain informations about MOS.

    Args:
        mos (str): MOS version (optional)
        Class (str): Class name (optional)
    Returns:
        str: JSON string containing all information about available classes. 
    """
    conn_classinfo = psycopg2.connect("host=100.107.93.75 dbname=Chatbot_db user=n8n_user password=n8n_pass")    
    with conn_classinfo:
        with conn_classinfo.cursor() as curs:
            curs.execute("SELECT * FROM public.thong_tin_lop_hoc")
            rows = curs.fetchall()
    conn_classinfo.close()
    class_info = []
    for info in rows:
        class_info.append({
            "mos_version": info[0],
            "language": info[1], 
            "class_name": info[2]
        })
    return json.dumps(class_info, ensure_ascii=False, indent=2)

@tool
def rag_search(text):
    """
    Queries the RAG (Retrieval-Augmented Generation) system to retrieve and generate an answer based on the input question.
    Its contain some information about MOS course in Tinz Center.

    Args:
        text (str): The input question or query to be searched.
    Returns:
        str: JSON string containing the generated answer based on retrieved relevant information with rank of similarity.
    """
    conn_vectorstore = psycopg2.connect("host=100.107.93.75 dbname=Chatbot_db user=n8n_user password=n8n_pass")
    new_embedding = embeddings.embed_query(text)
    with conn_vectorstore:
        with conn_vectorstore.cursor() as curs:
            curs = conn_vectorstore.cursor()
            curs.execute("""SELECT id, content FROM vector_store
                        ORDER BY embedding <-> %s::vector
                        LIMIT 3
                        """, (new_embedding,)
                        )
    results = curs.fetchall()
    conn_vectorstore.close()
    rag_answer = []
    for i in range(len(results)):
        rag_answer.append({
            "rank": i + 1,
            "id": results[i][0],
            "content": results[i][1] if len(results[i]) > 1 else ""
        })
    return json.dumps(rag_answer, ensure_ascii=False, indent=2)

from langgraph.types import interrupt, Command
import requests

@tool
def human_assistance(question: str) -> str:
    """
    Request human assistance when you can not get the information needed to answer. This tool have all information, and use this tool when the other tools can not provide needed information.
    Arg(str): 
        The question from user, which you can not answered.
    Return:
        str: The answer from human assistant.
    """
    webhook = "https://n8n-TinZ.aipencil.name.vn/webhook/human-in-the-loop"
    post = requests.post(url = webhook, params = {"question": question})  
    sale_answer = post.text
    knowledge_enriching(question, sale_answer)
    return sale_answer

from langchain_ollama import OllamaEmbeddings
import psycopg2
def knowledge_enriching(question:str, answer:str):
    embedding = OllamaEmbeddings(model = "bge-m3:latest")
    content = f"CÂU HỎI: {question} CÂU TRẢ LỜI: {answer}"
    chunk = embedding.embed_query(content)

    connection = psycopg2.connect("host=100.107.93.75 dbname=Chatbot_db user=n8n_user password=n8n_pass") 
    cursor = connection.cursor()
    cursor.execute("INSERT INTO vector_store (content, embedding) VALUES (%s, %s)", (content, chunk))

    connection.commit()
    cursor.close()
    connection.close()
