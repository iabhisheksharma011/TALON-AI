# 🦙 Ollama Offline Chat — RAG-powered chat with local LLMs

A sleek, terminal-style chat app that connects to your local Ollama instance.
Features streaming responses, persistent chat history, multi-file RAG with caching,
and a clean dark UI — all in a single Python project, no cloud required.

---

## Features

| Feature | Details |
|---|---|
| 🦙 Ollama integration | Auto-detects all locally installed models |
| 💬 Persistent chat history | SQLite-backed sessions, survives restarts |
| 📄 Multi-file RAG | Upload PDFs, TXT, MD, CSV, JSON, code files |
| ⚡ File caching | Already-indexed files load instantly from cache |
| 🔍 TF-IDF retrieval | Pure Python, no external vector DB needed |
| 🌊 Streaming responses | Tokens stream as they're generated |
| 🗂️ Multiple sessions | Rename, switch, delete chat sessions |

---

## Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Start the Ollama service
   ollama serve

   # Pull at least one model (in a new terminal)
   ollama pull llama3.2       # ~2GB, recommended for most machines
   ollama pull mistral        # alternative
   ollama pull phi3           # lightweight option
   ```

---

## Setup & Run

```bash
# 1. Navigate to the project folder
cd ollama-chat

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the app
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## How to Use

### Chat
1. Click **New Chat** in the sidebar
2. Select a model from the dropdown (top right of chat)
3. Type your message and press **Enter** to send

### Upload Documents
1. In the right panel, click the upload zone or drag files in
2. Supported: `.pdf`, `.txt`, `.md`, `.csv`, `.json`, `.py`, `.js`, `.ts`, `.html`, `.yaml`, `.log`
3. Files are chunked and indexed automatically
4. **Already-indexed files are loaded instantly from cache** on the next run

### RAG Toggle
- Use the **Docs ON/OFF** toggle (top bar) to enable/disable document context per message
- When ON, the top-5 most relevant chunks are injected into each prompt

### Chat Memory
- Every message is saved to SQLite (`cache/chat_memory.db`)
- Sessions persist across app restarts
- Click a session in the sidebar to resume it

---

## Project Structure

```
ollama-chat/
├── app.py              # Flask server + API routes
├── ollama_client.py    # Thin wrapper around Ollama HTTP API
├── rag_engine.py       # File indexing, chunking, TF-IDF retrieval, caching
├── chat_memory.py      # SQLite-backed session and message storage
├── requirements.txt
├── templates/
│   └── index.html      # Single-file frontend (HTML + CSS + JS)
├── uploads/            # Uploaded files stored here
└── cache/
    ├── rag.db          # File chunks & metadata (SQLite)
    └── chat_memory.db  # Chat sessions & messages (SQLite)
```

---

## Architecture Notes

- **No external vector DB**: RAG uses pure Python TF-IDF with cosine similarity
- **Caching**: File hashes are checked before re-indexing; unchanged files skip processing entirely
- **Streaming**: Uses Server-Sent Events (SSE) for real-time token streaming
- **Context window**: Last 20 messages + top-5 retrieved chunks per query
- **Chunk size**: 600 chars with 100-char overlap for good retrieval granularity
