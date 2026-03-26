# QA Pipeline Frontend

A professional, dark-themed Streamlit interface for querying the tourism QA pipeline with support for multiple retrieval modes.

## Features

✨ **Dark-themed UI** — Modern ChatGPT-like interface with a clean, professional design  
🎯 **Multiple Modes** — Select from `none`, `vectordb`, `kg`, or `hybrid` retrieval modes  
💬 **Session History** — View all questions and answers from the current session (clears on restart)  
⚡ **Real-time Processing** — Instantly see answers from the Test LLM component  
📱 **Responsive Design** — Works seamlessly on different screen sizes  

## Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment activated (from project root)

### Installation

The frontend uses only Streamlit, which should already be installed in your main project environment. If not:

```bash
pip install streamlit==1.28.1
```

### Running the Frontend

From the project root directory:

```bash
streamlit run frontend/streamlit_app.py
```

This will:
1. Start the Streamlit server (default: http://localhost:8501)
2. Automatically open the browser to the interface
3. Show a clean, dark-themed chat interface

### Usage

1. **Type a question** in the text input field
   - Example: "What are the best beaches in Mangalore?"

2. **Select a retrieval mode** from the dropdown:
   - `none` — No retrieval context (baseline)
   - `vectordb` — Vector database only
   - `kg` — Knowledge Graph only
   - `hybrid` — Both vector and KG (recommended)

3. **Click "Send"** to submit the query
   - The pipeline processes the query through all selected components
   - The Test LLM generates and returns an answer

4. **View your chat history** — All Q&A pairs are displayed in the current session
   - Mode used is shown below each answer
   - Scroll up to review previous interactions

### Session History

- **During Session**: All questions and answers are displayed and persist as you add more
- **On Restart**: Session history is cleared (Streamlit default behavior) - this is intentional

## Architecture

The frontend is completely decoupled from the pipeline logic:

- **No pipeline modifications** — Uses only the public `run_pipeline()` interface
- **Clean separations** — Frontend code isolated in `/frontend` folder
- **Sessions in session_state** — Streamlit's built-in state management for history

## File Structure

```
frontend/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Customization

To customize the appearance:

1. **Colors** — Edit the CSS in the `<style>` section of `streamlit_app.py`
   - `#10a37f` — Primary green color (user messages)
   - `#2d2d2d` — Secondary gray (assistant messages)
   - `#0d0d0d` — Background dark

2. **Layout** — Modify column ratios in the input area:
   - Currently: 3 (query input) : 1 (mode) : 0.5 (send button)

3. **Default Mode** — Change the default selected mode by editing:
   ```python
   index=3,  # 0=none, 1=vectordb, 2=kg, 3=hybrid
   ```

## Troubleshooting

### "ModuleNotFoundError: No module named 'backend'"

**Solution**: Make sure you're running from the project root directory:
```bash
cd /Users/sammy/Desktop/Coding/Major_Project/Major_Project
streamlit run frontend/streamlit_app.py
```

### "Connection refused" or "Pipeline not responding"

**Solution**: Ensure all backend services are configured:
- Check `.env` file has all required keys (GEMINI_API_KEY, Pinecone, Neo4j credentials)
- Vector DB and KG uploader have been run (sample data loaded)

### "No results returned"

**Solution**: This is normal for `kg` mode if the KG hasn't been populated. Try:
- `hybrid` mode — falls back to vector DB if KG is empty
- Run the KG uploader first: `python backend/KnowledgeGraph/main.py`

## Environment Variables

Make sure `backend/.env` contains:
- `GEMINI_API_KEY` — Google Gemini API key
- `VECTORDB_PINECONE_API_KEY` — Pinecone API key
- `KG_NEO4J_URI`, `KG_NEO4J_USERNAME`, `KG_NEO4J_PASSWORD` — Neo4j credentials

## Notes

- The frontend displays **only the answer** from the Test LLM component
- Full context (vectors, triples, prompt) is available in the pipeline but not displayed for a clean UI
- Each question creates a new pipeline run independently
- No data is persisted between server restarts (by design)

## Performance

Typical response times:
- `none` mode: < 1 second
- `vectordb` mode: 2-5 seconds (includes Pinecone retrieval + reranking)
- `kg` mode: 2-5 seconds (includes Neo4j query + reranking)
- `hybrid` mode: 3-8 seconds (parallel retrieval from both sources)
