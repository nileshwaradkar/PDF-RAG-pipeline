
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore


def create_embeddings(model):
    embeddings = OllamaEmbeddings(model=model)
    return embeddings

def create_vector_store(embeddings, all_splits):
    vector_store = InMemoryVectorStore(embedding=embeddings)
    _ = vector_store.add_documents(documents=all_splits)
    return vector_store
