import os
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables from your .env file
load_dotenv()

from app.core.rag import RAG
from app.models.schemas import QueryRequest, QueryResponse

# --- Configuration for File Paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "vector_store", "faiss.index")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "vector_store", "chunk_mapping.pkl")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the RAG system on application startup.
    The RAG instance is stored in the app's state for access across requests.
    """
    print("Application startup: Initializing RAG system...")
    
    openai_api_key = os.getenv("OPEN_AI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPEN_AI_API_KEY environment variable not set.")

    # Storing the RAG instance in `app.state` is the recommended FastAPI pattern
    # for sharing objects (like models, DB connections) across requests.
    app.state.rag_system = RAG(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        index_path=INDEX_PATH,
        mapping_path=MAPPING_PATH,
        llm_key=openai_api_key
    )
    print("RAG system initialized successfully.")

    yield  # Application is running

    # Optional: Cleanup actions (if needed)
    print("Application shutdown.")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG API",
    description="An API for Retrieval-Augmented Generation using FastAPI and OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.post("/query")
async def query_rag(request: Request, query: QueryRequest):
    """
    Endpoint to handle RAG queries. It retrieves the initialized RAG system
    from the app state and uses it to generate a response.
    """
    # Access the RAG system from the application state.
    rag_system: RAG = request.app.state.rag_system
    
    try:
        response = rag_system.get_rag_response(query)
        print(response.answer)
        return response.answer
    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@app.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint to confirm the service is running.
    """
    return {"status": "ok"}
