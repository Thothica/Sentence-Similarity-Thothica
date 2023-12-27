import streamlit as st
import os
import json
from llama_index import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def create_index():
    index = load_index_from_storage(storage_context = StorageContext.from_defaults(
                docstore = SimpleDocumentStore.from_persist_dir(persist_dir = "storage"),
                vector_store = FaissVectorStore.from_persist_dir(persist_dir = "storage"),
                index_store = SimpleIndexStore.from_persist_dir(persist_dir = "storage"),
            ))
    with open("text_order.json", "r") as f:
        text_order = json.load(f)
    return index, text_order

st.title("Thothica Sentence Similarity Prototype")

query = st.text_input(label = 'Please enter your query - ', value = 'What causes ocean acidification?')
top_k = st.number_input(label = 'Top k - ', min_value = 2, max_value = 125, value = 3)
unique_top_k = st.checkbox(label = "Do you want top k to be from unique documents [Note - Please use lower values of k when ticked]", value = True)

index, text_order = create_index()

if query and top_k:
    response = []
    unique_url = []
    if unique_top_k:
        multiplier = 1
        while len(response) != top_k:
            for i in index.as_retriever(retriever_mode = 'embedding', similarity_top_k = int(top_k * multiplier)).retrieve(query):
                if i.node.metadata["Title_URL"] in unique_url:
                    continue
                else:
                    response.append({
                            'Text' : i.get_text(),
                            'Score' : i.get_score(),
                            'Title_URL' : i.node.metadata['Title_URL'],
                            'Author' : i.node.metadata['Author'],
                            'Publisher' : i.node.metadata['Publisher'],
                            'Title_URL' : i.node.metadata['Title_URL'],
                            'Type' : i.node.metadata['Type'],
                            'Title' : i.node.metadata['Title']
                        })
                    unique_url.append(i.node.metadata["Title_URL"])
            multiplier += 1
    else:
        for i in index.as_retriever(retriever_mode = 'embedding', similarity_top_k = int(top_k)).retrieve(query):
            response.append({
                    'Text' : i.get_text(),
                    'Score' : i.get_score(),
                    'Title_URL' : i.node.metadata['Title_URL'],
                    'Author' : i.node.metadata['Author'],
                    'Publisher' : i.node.metadata['Publisher'],
                    'Title_URL' : i.node.metadata['Title_URL'],
                    'Type' : i.node.metadata['Type'],
                    'Title' : i.node.metadata['Title']
                })
    st.divider()
    for n, i in enumerate(response):
        title = i["Title"]
        text_arr = text_order[title]
        text_idx = text_arr.index(i["Text"])
        if text_idx == 0:
            prev = ""
        else:
            prev = text_arr[text_idx - 1]
        if text_idx == len(text_arr) - 1:
            nxt = ""
        else:
            nxt = text_arr[text_idx + 1]
        st.subheader(title)
        st.markdown(f"{prev} :orange[{i['Text']}] {nxt}")
        st.write(f"Source - {i['Title_URL']}")
        st.divider()