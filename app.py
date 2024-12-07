import os
import time  # Importing the time module for measuring execution time
from dotenv import load_dotenv
from llama_index.core import GPTVectorStoreIndex
from llama_index.llms.groq import Groq
from src.utils.data import models_list
from src.utils.preprocess_data import preprocess_documents
from src.utils.generate_embedding import InstructorEmbeddings

# Load environment variables
load_dotenv()

# Function 1: Load and preprocess documents
def load_and_preprocess_documents(data_source):
    return preprocess_documents(data_source)

# Function 2: Build the index
def build_index(documents, embed_model):
    return GPTVectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Function 3: Retrieve documents
def retrieve_documents(index, query):
    retriever = index.as_retriever()
    return retriever.retrieve(query)

# Function 4: Query the LLM with context
def query_llm_with_context(llm, context, user_query):
    rag_query = f"Context: {context}\n\nQuery: {user_query}"
    return llm.complete(rag_query)

# Function 5: RAG pipeline
def rag_pipeline(data_source, embed_model, llm, user_query):
    # Step 1: Load and preprocess documents
    start_time = time.time()
    documents = load_and_preprocess_documents(data_source)
    print(f"Preprocessing Time: {time.time() - start_time:.2f} seconds")

    # Step 2: Build the index
    start_time = time.time()
    index = build_index(documents, embed_model)
    print(f"Index Building Time: {time.time() - start_time:.2f} seconds")

    # Step 3: Retrieve documents
    start_time = time.time()
    retrieved_docs = retrieve_documents(index, user_query)
    retrieved_context = "\n".join([doc.text for doc in retrieved_docs])
    print(f"Retrieval Time: {time.time() - start_time:.2f} seconds")

    # Step 4: Query the LLM with retrieved context
    start_time = time.time()
    response = query_llm_with_context(llm, retrieved_context, user_query)
    print(f"LLM Query Time: {time.time() - start_time:.2f} seconds")

    return response

# Main script
if __name__ == "__main__":
    # Record total start time
    total_start_time = time.time()

    # Define API key and query
    api_key = os.getenv("groq_token")
    query = "Explain the importance of low latency LLMs"

    # Initialize components
    embed_model = InstructorEmbeddings()
    llm = Groq(model="llama3-70b-8192", api_key=api_key)

    # Run the RAG pipeline
    response = rag_pipeline(
        data_source=models_list,
        embed_model=embed_model,
        llm=llm,
        user_query=query
    )

    # Print the final response
    print("\nRAG Response:")
    print(response)

    # Record total execution time
    total_end_time = time.time()
    print(f"\nTotal Execution Time: {total_end_time - total_start_time:.2f} seconds")
