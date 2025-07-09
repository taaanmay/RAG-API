# RAG-API

## Instructions

To generate embeddings and build the FAISS vector store, run the following command:

```bash
python -m app.ingest
```

This will process the documents in your configured data directory, generate embeddings using the specified model, and store them in a FAISS index for efficient retrieval.

**Details:**
- The `ingest.py` script loads all documents from the `data` folder.
- It uses the embedding model defined in your environment or configuration.
- The resulting FAISS index is saved to disk for use during API queries.
- If you add or update documents, rerun the command to refresh the index.
- Ensure your environment variables (such as API keys) are set before running the script.
- For advanced options, review the arguments and configuration at the top of `app/ingest.py`.

## Running the API Server

To start the API server using Uvicorn, run:

```bash
uvicorn app.main:app --reload
```

This will launch the server at `http://127.0.0.1:8000/`. The `--reload` flag enables automatic reloading on code changes, which is useful during development.

## Testing the API

You can test the `/query` endpoint with a `curl` command:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{"question": "What is the main topic of the document?"}'
```

This sends a POST request with your question in JSON format. The API will respond with an answer based on the indexed documents.

## Additional Information

- The API provides endpoints for querying the vector store and retrieving relevant information from your documents.
- Review the OpenAPI docs at `http://127.0.0.1:8000/docs` for interactive documentation and to explore available endpoints.
- Make sure the FAISS index has been generated before starting the server, otherwise queries may not return results.
- You can customize the API behavior and endpoints by modifying `app/main.py`.