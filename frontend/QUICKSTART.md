# Frontend Quick Start Guide

## One-Command Start

```bash
cd /Users/sammy/Desktop/Coding/Major_Project/Major_Project
streamlit run frontend/streamlit_app.py
```

The UI will automatically open in your browser at `http://localhost:8501`

## What You'll See

### Welcome Screen (First Load)
- Title: "💬 QA Pipeline"
- Subtitle: "Query your knowledge base with multiple retrieval modes"
- Empty chat area with instructions

### After First Query
- **User Message** (right side, green)
  - Your question in a chat bubble
  
- **Assistant Response** (left side, gray)
  - Answer from the Test LLM
  - Mode badge showing which retrieval mode was used

### Interface Components

1. **Query Input** (bottom)
   - Text field: "Ask a question about tourism in Karnataka..."
   - Full width input area

2. **Mode Selector** (bottom right)
   - Dropdown with 4 options:
     - `none` — No retrieval (baseline)
     - `vectordb` — Vector database only
     - `kg` — Knowledge Graph only
     - `hybrid` — Both (default, recommended)

3. **Send Button** (bottom right)
   - Green button with "Send" text
   - Disabled when input is empty
   - Shows spinner while processing

## Example Queries

Try these questions:

```
"What are the best beaches in Mangalore?"
  ↳ Mode: hybrid (best for general queries)

"How do I get to Malpe?"
  ↳ Mode: hybrid

"What activities are available in Udupi?"
  ↳ Mode: kg (structured queries work well with KG)

"Tell me about Karnataka"
  ↳ Mode: none (baseline, no context)
```

## How Chat History Works

- **Messages persist** during your session
- **Scroll up** to see previous Q&A pairs
- **Mode shown** under each assistant response
- **Cleared on restart** — Streamlit server restarts clear history (by design)

## Performance Tips

- **First response slower** (~3-8s) due to model loading
- **Subsequent queries faster** (~1-3s) as models are cached
- **Hybrid mode recommended** for best results (combines both sources)
- **KG mode** works only if KG has been populated with data

## If Something Goes Wrong

### "ModuleNotFoundError: No module named 'backend'"
```bash
# Make sure you're in the project root:
cd /Users/sammy/Desktop/Coding/Major_Project/Major_Project
streamlit run frontend/streamlit_app.py
```

### "Connection refused" or "No response from pipeline"
- Check `.env` file has all required keys
- Run vector DB uploader: `python vectordb_uploader/main.py`
- Run KG uploader: `python backend/KnowledgeGraph/main.py`

### "KG mode returns 'No KG data found'"
- This is normal if KG hasn't been populated
- Try `hybrid` or `vectordb` mode instead
- Or run: `python backend/KnowledgeGraph/main.py`

### Page not responsive or looks broken
- Try refreshing the page (F5/Cmd+R)
- Close and restart the server:
  ```bash
  # Ctrl+C to stop
  streamlit run frontend/streamlit_app.py
  ```

## Customizing the UI

Edit `frontend/streamlit_app.py` to customize:

### Change default mode:
```python
# Line ~200, change index parameter:
index=2,  # Change from 3 (hybrid) to another:
          # 0=none, 1=vectordb, 2=kg, 3=hybrid
```

### Change colors:
```python
# Line ~30-50, modify CSS:
primaryColor = "#10a37f"           # Green (user messages)
secondaryBackgroundColor = "#2d2d2d" # Gray (assistant messages)
backgroundColor = "#0d0d0d"         # Dark background
```

### Adjust layout ratios:
```python
# Line ~200, modify column sizes:
col1, col2, col3 = st.columns([3, 1, 0.5])
# Change [3, 1, 0.5] to adjust input/mode/button ratios
```

## Deployment Notes

For production deployment (e.g., AWS, DigitalOcean):

1. **Use environment variables** for all credentials (not .env)
2. **Set `streamlit config server.headless = true`**
3. **Use appropriate port** and firewall rules
4. **Enable HTTPS** in reverse proxy
5. **Scale backend services** separately

## Keyboard Shortcuts

- **Enter** — Send message (from input field)
- **Ctrl+L** (Streamlit) — Clear cache
- **Ctrl+C** — Stop server

## Support

For issues or questions:
1. Check `frontend/README.md` for detailed docs
2. Review troubleshooting in main `README.md`
3. Check backend logs for error details
