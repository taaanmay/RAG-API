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