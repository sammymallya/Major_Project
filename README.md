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

## Command Runbook

Run all commands from project root:

```bash
cd /Users/sammy/Desktop/Coding/Major_Project/Major_Project
```

Use the root venv interpreter in commands below:

```bash
./venv/bin/python
```

### 1) Run API and Whole Pipeline

Start FastAPI:

```bash
./venv/bin/python -m uvicorn backend.api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Run complete pipeline via API (`test_mode` can be `none`, `vectordb`, `kg`, `hybrid`):

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are some beaches in Mangalore?","test_mode":"hybrid"}'
```

### 2) Node-Level Test Commands

Query Structurer:

```bash
./venv/bin/python -m backend.scripts.test_query_structurer --query "beaches in mangalore" --output-kind both
```

Vector DB (component script):

```bash
./venv/bin/python -m backend.scripts.test_vectordb --query "beaches near Mangalore" --top-k 5
```

Vector DB (diagnostic node contact + retrieval check):

```bash
./venv/bin/python scripts/test_vectordb_node.py --query "beaches" --top-k 5
```

KG node:

```bash
./venv/bin/python -m backend.scripts.test_kg --question "Beaches in Mangalore"
```

Vector reranker:

```bash
./venv/bin/python -m backend.scripts.test_reranker
```

KG reranker:

```bash
./venv/bin/python -m backend.scripts.test_kg_reranker --question "Beaches in Mangalore"
```

Prompt generator:

```bash
./venv/bin/python -m backend.scripts.test_prompt_generator
```

Test LLM:

```bash
./venv/bin/python -m backend.scripts.test_test_llm
```

### 3) Vector DB Dataset Upload

Dry run:

```bash
./venv/bin/python vectordb_uploader/main.py --file vectordb_uploader/dataset/vectordb_dataset.json --dry-run
```

Upload dataset:

```bash
./venv/bin/python vectordb_uploader/main.py --file vectordb_uploader/dataset/vectordb_dataset.json
```

Notes:
- Uploader uses only `text` from each record.
- IDs are auto-generated sequentially (default `tourism_001...`).
- Batch size is capped to Pinecone integrated-embedding max (96).

### 4) Run Evaluation Engine on test_data.json

Default run (single execution per query per mode, no calibration):

```bash
./venv/bin/python run_evaluation.py --test-data test_data.json --output evaluation_results.json --csv-prefix evaluation_results
```

Optional calibration-enabled run:

```bash
./venv/bin/python run_evaluation.py --test-data test_data.json --output evaluation_results.json --csv-prefix evaluation_results --enable-calibration
```

Generated files:
- `evaluation_results.json`
- `evaluation_results_detail.csv`
- `evaluation_results_summary.csv`

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

Enjoy exploring Karnataka tourism!