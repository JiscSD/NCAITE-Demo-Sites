from helper import load_mistral_api_key

api_key, dlai_endpoint = load_mistral_api_key(ret_key=True)

import panel as pn

pn.extension()


def run_mistral(contents, user, chat_interface):
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    messages = [ChatMessage(role="user", content=contents)]
    chat_response = client.chat(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content


chat_interface = pn.chat.ChatInterface(callback=run_mistral, callback_user="Mistral")

chat_interface


import requests
from bs4 import BeautifulSoup
import re

response = requests.get(
    "https://beta.jiscuni.ai.alpha.jisc.ac.uk/"
)
html_doc = response.text
soup = BeautifulSoup(html_doc, "html.parser")
tag = soup.find("div", re.compile("^prose--styled"))
text = tag.text
print(text)

# Optionally save this text into a file.
file_name = "AI_greenhouse_gas.txt"
with open(file_name, "w") as file:
    file.write(text)

import numpy as np
import faiss

client = MistralClient(
    api_key=os.getenv("MISTRAL_API_KEY"),
    endpoint=os.getenv("DLAI_MISTRAL_API_ENDPOINT"),
)

prompt = """
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}   
Answer:
"""

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(model="mistral-embed", input=input)
    return embeddings_batch_response.data[0].embedding

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [ChatMessage(role="user", content=user_message)]
    chat_response = client.chat(model=model, messages=messages)
    return chat_response.choices[0].message.content

def answer_question(question, user, instance):
    text = file_input.value.decode("utf-8")

    # split document into chunks
    chunk_size = 2048
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    # load into a vector database
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)
    # create embeddings for a question
    question_embeddings = np.array([get_text_embedding(question)])
    # retrieve similar chunks from the vector database
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    # generate response based on the retrieved relevant text chunks
    response = run_mistral(
        prompt.format(retrieved_chunk=retrieved_chunk, question=question)
    )
    return response

file_input = pn.widgets.FileInput(accept=".txt", value="", height=50)

chat_interface = pn.chat.ChatInterface(
    callback=answer_question,
    callback_user="Mistral",
    header=pn.Row(file_input, "### Upload a text file to chat with it!"),
)
chat_interface.send(
    "Send a message to get a reply from Mistral!", user="System", respond=False
)
chat_interface
