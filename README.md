# Hybrid Memory Augmented LLM for Karnataka Tourism

## Overview

This backend system implements a hybrid Retrieval-Augmented Generation (RAG) pipeline for Karnataka tourism queries, with a focus on Mangalore and Udupi regions. It combines vector database similarity search, knowledge graph structured queries, and cross-encoder reranking to retrieve relevant context, build prompts, and generate answers using a placeholder LLM.

The system is designed for scalability, with modular components that can be tested individually or integrated via a FastAPI server.

## Architecture

- **API Layer** (`backend/api/`): FastAPI application with a `/query` endpoint. Accepts JSON requests with a natural language query and routing mode (`vectordb`, `kg`, `hybrid`, `none`).
- **Orchestration** (`backend/services/orchestration.py`): Core pipeline logic. Routes queries based on mode, retrieves data, reranks, builds prompts, and calls the placeholder LLM.
- **Query Structurer** (`backend/query_structurer/`): Uses Google Gemini API to convert natural language into structured queries (semantic search string + Cypher query for Neo4j).
- **Vector Database** (`backend/vector_db/`): Pinecone integration for embedding-based similarity search using Sentence Transformers.
- **Knowledge Graph** (`backend/kg/`): Neo4j client for executing Cypher queries on tourism entities (places, cities, districts, states).
- **Rerankers** (`backend/reranker/`, `backend/kg_reranker/`): Cross-encoder models to refine and rank retrieved results.
- **DTOs** (`backend/dto/`): Pydantic models for request/response validation.
- **Scripts** (`backend/scripts/`): Individual test scripts for each component.

Data flow: Query → Structurer → Retrieve (Vector/KG) → Rerank → Prompt → LLM → Response.

## Prerequisites

- **Python**: 3.8 or higher.
- **Accounts/Services**:
  - Neo4j Aura (cloud Neo4j instance for knowledge graph).
  - Pinecone (vector database).
  - Google Gemini API (for query structuring).
- **Tools**: Virtual environment (venv), curl/Postman for API testing.
- **OS**: macOS/Linux/Windows (tested on macOS).

## Setup

1. **Navigate to Project Directory**:
   ```bash
   cd /Users/sammy/Desktop/Coding/Major_Project/Major_Project
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Configure Environment Variables**:
   Create `backend/.env` with the following (replace with your actual keys):
   ```
   # Neo4j (Knowledge Graph)
   KG_NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   KG_NEO4J_USERNAME=neo4j
   KG_NEO4J_PASSWORD=your_password

   # Pinecone (Vector Database)
   VECTORDB_PINECONE_API_KEY=your_pinecone_api_key
   VECTORDB_PINECONE_INDEX_NAME=your_index_name
   VECTORDB_PINECONE_ENVIRONMENT=your_environment

   # Google Gemini (Query Structurer)
   QUERY_STRUCTURER_API_KEY=your_gemini_api_key
   ```
   - Obtain keys from respective service dashboards.
   - Ensure Neo4j has tourism data loaded (places with types like "Beach").

## Running the System

### Start the FastAPI Server

Run the server with auto-reload for development:
```bash
uvicorn backend.api.main:app --reload
```

- **Server URL**: `http://127.0.0.1:8000`
- **API Documentation**: Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI.
- **Health Check**: `GET http://127.0.0.1:8000/health` returns `{"status": "ok"}`.

### Test the API

Use curl to send a POST request:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Beaches in Mangalore", "test_mode": "kg"}'
```

**Expected Response** (example for `kg` mode):
```json
{
  "answer": "[Placeholder answer for prompt length=150]",
  "test_mode": "kg",
  "context_used": {
    "kg_triples_count": 1,
    "kg_snippet": "Someshwar Beach --LOCATED_IN--> Mangalore"
  }
}
```

- `test_mode` options: `"vectordb"` (vector only), `"kg"` (KG only), `"hybrid"` (both), `"none"` (no retrieval).
- On failure (e.g., no data), `answer` will be an error message, and `context_used` may be null or partial.

## Testing Individual Components

All scripts run from the project root. They load `backend/.env` and test specific components. Use these to debug or verify setup.

1. **Test Knowledge Graph (Neo4j)**:
   ```bash
   python -m backend.scripts.test_kg --question "Beaches in Mangalore"
   ```
   - Checks Neo4j connection.
   - Extracts entities, builds Cypher, runs query, formats answer.
   - Output: Human-readable answer or error.

2. **Test Vector Database (Pinecone)**:
   ```bash
   python -m backend.scripts.test_vectordb --query "beaches near Mangalore" --top-k 5
   ```
   - Embeds query, fetches top results from Pinecone.
   - Output: List of results with IDs, scores, text.

3. **Test Query Structurer (Gemini)**:
   ```bash
   python -m backend.scripts.test_query_structurer --query "beaches in mangalore" --output-kind kg_only
   ```
   - Generates Cypher query.
   - Output: StructuredQuery with `cypher_query`.

   For both semantic and Cypher:
   ```bash
   python -m backend.scripts.test_query_structurer --query "beaches in mangalore" --output-kind both
   ```
   - Output: Both `semantic_search_query` and `cypher_query`.

4. **Test KG Reranker**:
   ```bash
   python -m backend.scripts.test_kg_reranker --question "Beaches in Mangalore"
   ```
   - Fetches KG triples, reranks top 1.
   - Output: Reranked triple details.

5. **Test Vector Reranker**:
   ```bash
   # Similar to KG reranker, but for vector results
   python -m backend.scripts.test_reranker  # (Assuming script exists; adjust if needed)
   ```

Run with `--verbose` for debug logs if issues arise.

## Usage Examples

- **Vector-Only Search**: `{"query": "best restaurants in Udupi", "test_mode": "vectordb"}` → Retrieves similar documents.
- **KG Facts**: `{"query": "places in Mangalore", "test_mode": "kg"}` → Structured triples from graph.
- **Hybrid**: `{"query": "beaches and hotels in Mangalore", "test_mode": "hybrid"}` → Combines vector snippets and KG triples.
- **Direct Query**: `{"query": "What is tourism?", "test_mode": "none"}` → No retrieval, just query as prompt.

## Troubleshooting

- **Import Errors**: Ensure venv is activated and all packages from `requirements.txt` are installed.
- **API Key Issues**: Verify `.env` values. Gemini may show deprecation warnings—consider upgrading to `google-genai`.
- **No Data Returned**: Check Neo4j/Pinecone for loaded data. Test individual scripts first.
- **Server Won't Start**: Ensure ports are free; check for missing env vars.
- **Query Structurer Fails**: Gemini API may be rate-limited or invalid. Test with simple queries.
- **Logs**: Add `logging.basicConfig(level=logging.DEBUG)` in scripts for more details.
- **Common Errors**:
  - `ModuleNotFoundError`: Reinstall dependencies.
  - `ConnectionError`: Check internet and service credentials.
  - Empty responses: Data not loaded in databases.

For further help, check component logs or raise issues with output examples.

## Contributing

- Test components individually before integration.
- Update `.env` template for new keys.
- Ensure Cypher queries return triples for KG compatibility.

Enjoy exploring Karnataka tourism! 🏖️</content>
<parameter name="filePath">/Users/sammy/Desktop/Coding/Major_Project/Major_Project/README.md