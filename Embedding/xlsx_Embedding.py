#XLSE ONLY !!!
#XLSE ONLY !!!
#XLSE ONLY !!!
import re
from langchain_community.document_loaders import UnstructuredExcelLoader
loader = UnstructuredExcelLoader("./Tinz.xlsx", mode="paged")
docs = loader.load()

questions_and_anwsers = docs[0].metadata["text_as_html"].strip("<table><tr><td>Câu hỏi</td><td>Câu trả lời</td><td>Lưu ý")
questions_and_anwsers = re.split(r"</td>(?:<td/>)?</tr><tr><td>", questions_and_anwsers)
texts_processed = [] 

print("Input processing...")
for item in questions_and_anwsers:
    new_item = f"CÂU HỎI: {item.replace('</td><td>', ' CÂU TRẢ LỜI: ')}"
    if "\\n" in new_item:
        new_item = new_item.replace("\\n", " ")
    texts_processed.append(new_item)

from langchain_ollama import OllamaEmbeddings
embedding = OllamaEmbeddings(model = "bge-m3:latest") #1024 dim
embedding_list = [] 

print("Embedding...")
for text in texts_processed:
    embedding_list.append(embedding.embed_query(text))

import psycopg2
connection = psycopg2.connect("host=100.107.93.75 dbname=Chatbot_db user=n8n_user password=n8n_pass") 
cursor = connection.cursor()
print("Storing vectors...")
for i in range(len(embedding_list)):
    embedding = embedding_list[i]
    content = texts_processed[i]
    cursor.execute("INSERT INTO vector_store (content, embedding) VALUES (%s, %s)", (content, embedding))

connection.commit()
cursor.close()
connection.close()

print("Success!")
