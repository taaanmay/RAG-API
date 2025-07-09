import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from app.models.schemas import QueryRequest, QueryResponse

class RAG:
    def __init__(self, embedding_model_name: str, index_path: str, mapping_path: str, llm_key: str):
        """
        Initializes the RAG system by loading all necessary models and data files.
        This is done once to ensure efficient request handling.
        """
        print("Initializing RAG System...")
        
        # --- Load Models and Data ---
        # 1. Load the sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # 2. Load the FAISS index from disk
        self.index = faiss.read_index(index_path)
        
        # 3. Load the chunk mapping dictionary
        with open(mapping_path, 'rb') as f:
            self.chunk_mapping = pickle.load(f)
            
        # 4. Initialize the OpenAI client
        self.openai_client = OpenAI(api_key=llm_key)
        
        print("RAG System Initialized Successfully.")

    def get_rag_response(self, request: QueryRequest) -> QueryResponse:
        """
        Processes the query request, finds relevant context, and generates a response.
        """
        question = request.question
        top_k = request.top_k

        # 1. Embed the user's question
        #    We encode the question and ensure it's a numpy array of type float32.
        question_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        question_embedding = question_embedding.astype('float32').reshape(1, -1)

        # 2. Search the FAISS index for the most relevant chunk IDs
        #    D: distances, I: indices (our chunk IDs)
        distances, indices = self.index.search(question_embedding, top_k)

        # 3. Retrieve the actual text chunks using the retrieved IDs
        context_chunks = [self.chunk_mapping[i] for i in indices[0]]

        # 4. Construct the augmented prompt for the LLM
        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = f"""
        You are an expert assistant. Use the following pieces of context to answer the user's question.
        If you don't know the answer from the context provided, just say that you don't know. Do not make up an answer.

        Context:
        {context_str}

        Question: {question}
        
        Answer:
        """

        # 5. Make the API call to OpenAI
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            answer = "Sorry, I encountered an error while generating a response."

        # 6. Return the final answer and the context chunks that were used
        return QueryResponse(answer=answer, context=context_chunks)

