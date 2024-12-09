from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from llama_index.llms.groq import Groq
from src.utils.data import models_list
from src.utils.generate_embedding import InstructorEmbeddings
from src.models.rag_utils import load_and_preprocess_documents, build_index, save_index, load_index, rag_pipeline

app = FastAPI()

# Directory for HTML templates
templates = Jinja2Templates(directory="src/templates")

INDEX_STORAGE_PATH = "index_storage"

# Initialize components
api_key = os.getenv("groq_token")
llm = Groq(model="llama3-70b-8192", api_key=api_key)

if not os.path.exists(INDEX_STORAGE_PATH):
    print("Building and saving the index...")
    embed_model = InstructorEmbeddings()
    documents = load_and_preprocess_documents(models_list)
    index = build_index(documents, embed_model)
    save_index(index, INDEX_STORAGE_PATH)
else:
    print("Loading prebuilt index...")
    embed_model = InstructorEmbeddings()
    index = load_index(INDEX_STORAGE_PATH, embed_model)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def query_form(request: Request):
    """
    Display a form for entering the query.
    """
    return templates.TemplateResponse("query_form.html", {"request": request})

@app.post("/rag-response", response_class=HTMLResponse)
async def rag_response(request: Request, user_query: str = Form(...)):
    """
    Process the query and return the RAG pipeline response.
    """
    response = rag_pipeline(index, llm, user_query)
    return templates.TemplateResponse("query_response.html", {"request": request, "query": user_query, "response": response})
