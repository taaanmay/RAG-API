import os
import faiss
import pickle
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root directory
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_base")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss.index")
MAPPING_FILE = os.path.join(VECTOR_STORE_DIR, "chunk_mapping.pkl")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- 1. Load Documents ---
def load_documents(directory):
    """Loads all .txt files from the specified directory."""
    print(f"Loading documents from {os.path.abspath(directory)}...")
    all_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_texts.append(file.read())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    print(f"Loaded {len(all_texts)} documents.")
    return all_texts

# --- 2. Chunk Texts ---
def chunk_texts(texts):
    """Splits the loaded texts into smaller chunks."""
    print("Splitting texts into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, # Reduced overlap slightly for efficiency
        length_function=len
    )
    chunks = text_splitter.split_text("\n\n".join(texts))
    print(f"Created {len(chunks)} chunks.")
    return chunks

# --- 3. Create Embeddings ---
def create_embeddings(chunks):
    """Creates vector embeddings for the text chunks."""
    print("Loading embedding model and creating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    # The encode method directly returns numpy arrays, which is what FAISS needs.
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"Embeddings created with shape: {embeddings.shape}")
    return embeddings

# --- 4. Build and Save FAISS Index ---
def build_and_save_faiss_index(embeddings, chunks):
    """Builds a FAISS index and saves it along with the chunk mapping."""
    print("Building and saving FAISS index...")
    
    # Ensure the output directory exists
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # Get the dimension of the embeddings
    embedding_dimension = embeddings.shape[1]
    
    # Using IndexFlatL2 is a good starting point for exact search.
    # IndexIDMap allows us to map the vector's position back to our original chunk ID.
    index = faiss.IndexFlatL2(embedding_dimension)
    index = faiss.IndexIDMap(index)

    # We create an array of IDs corresponding to the chunk indices
    ids = np.array(range(len(chunks)))

    # Add the embeddings with their corresponding IDs to the index
    index.add_with_ids(embeddings.astype('float32'), ids)

    # Save the FAISS index to disk
    faiss.write_index(index, INDEX_FILE)

    # Save the mapping from ID to chunk text
    chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
    with open(MAPPING_FILE, 'wb') as f:
        pickle.dump(chunk_mapping, f)
    
    print(f"Index and mapping saved to {os.path.abspath(VECTOR_STORE_DIR)}")

# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Load
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    
    if not documents:
        print("No documents found. Please add .txt files to the data/knowledge_base directory.")
    else:
        # Step 2: Chunk
        chunks = chunk_texts(documents)
        
        # Step 3: Embed
        embeddings = create_embeddings(chunks)
        
        # Step 4: Index and Save
        build_and_save_faiss_index(embeddings, chunks)
        
        print("\n--- Ingestion Complete! ---")
