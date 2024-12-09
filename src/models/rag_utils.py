import os
from dotenv import load_dotenv
from llama_index.core import GPTVectorStoreIndex, StorageContext, load_index_from_storage
from src.utils.preprocess_data import preprocess_documents
from src.utils.generate_embedding import InstructorEmbeddings

load_dotenv()

INDEX_STORAGE_PATH = "index_storage"

def load_and_preprocess_documents(data_source):
    return preprocess_documents(data_source)

def build_index(documents, embed_model):
    return GPTVectorStoreIndex.from_documents(documents, embed_model=embed_model)

def save_index(index, storage_path):
    index.storage_context.persist(storage_path)

def load_index(storage_path, embed_model=None):
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    if embed_model is None:
        embed_model = InstructorEmbeddings() 
    return load_index_from_storage(storage_context, embed_model=embed_model)

def retrieve_documents(index, query):
    retriever = index.as_retriever()
    return retriever.retrieve(query)

def query_llm_with_context(llm, context, user_query):
    rag_query = f"Context: {context}\n\nQuery: {user_query}"
    return llm.complete(rag_query)

def rag_pipeline(index, llm, user_query):
    retrieved_docs = retrieve_documents(index, user_query)
    retrieved_context = "\n".join([doc.text for doc in retrieved_docs])
    return query_llm_with_context(llm, retrieved_context, user_query)