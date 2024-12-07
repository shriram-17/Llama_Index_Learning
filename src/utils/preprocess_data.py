from llama_index.core import Document

def preprocess_documents(models_list):  
    documents = []
    for item in models_list:
        content = f"Name: {item['name']}\nDescription: {item['description']}"
        documents.append(Document(text=content))  
    return documents