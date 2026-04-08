"""
Flask backend for Ollama Chat.
Run with: python app.py
"""

import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context

import ollama_client as ollama
import chat_memory as memory
from rag_engine import rag_index

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts",
    ".html", ".htm", ".xml", ".yaml", ".yml", ".rst", ".log"
}


# ── Serve Frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ── Ollama ────────────────────────────────────────────────────────────────────

@app.route("/api/ollama/status")
def ollama_status():
    return jsonify({"running": ollama.is_running()})


@app.route("/api/ollama/models")
def get_models():
    try:
        models = ollama.get_models()
        return jsonify({"models": models})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503


# ── Sessions ─────────────────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    return jsonify(memory.list_sessions())


@app.route("/api/sessions", methods=["POST"])
def create_session():
    data = request.json or {}
    sid = memory.create_session(name=data.get("name"), model=data.get("model", ""))
    return jsonify({"session_id": sid})


@app.route("/api/sessions/<int:sid>", methods=["DELETE"])
def delete_session(sid):
    memory.delete_session(sid)
    return jsonify({"ok": True})


@app.route("/api/sessions/<int:sid>/rename", methods=["PUT"])
def rename_session(sid):
    data = request.json or {}
    memory.rename_session(sid, data.get("name", "Untitled"))
    return jsonify({"ok": True})


@app.route("/api/sessions/<int:sid>/messages", methods=["GET"])
def get_messages(sid):
    return jsonify(memory.get_messages(sid, limit=200))


# ── Chat (streaming) ──────────────────────────────────────────────────────────

@app.route("/api/chat/<int:sid>", methods=["POST"])
def chat(sid):
    data = request.json or {}
    user_msg = data.get("message", "").strip()
    model = data.get("model", "")
    use_rag = data.get("use_rag", True)

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    if not model:
        return jsonify({"error": "No model selected"}), 400

    # Build context from RAG if files are indexed
    context_chunks = []
    if use_rag:
        context_chunks = rag_index.retrieve(user_msg)

    # Persist user message
    memory.add_message(sid, "user", user_msg)
    memory.update_session_model(sid, model)

    # Build messages for LLM
    system_prompt = _build_system_prompt(context_chunks)
    history = memory.get_history_for_llm(sid, max_messages=20)

    # history already includes the user message we just added
    llm_messages = []
    if system_prompt:
        llm_messages.append({"role": "system", "content": system_prompt})
    llm_messages.extend(history)

    def generate():
        full_response = []
        try:
            for token in ollama.chat_stream(model, llm_messages):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Unexpected error: {e}'})}\n\n"
            return

        # Save assistant response
        assistant_text = "".join(full_response)
        if assistant_text:
            memory.add_message(sid, "assistant", assistant_text)

        sources = [{"filename": c["filename"], "score": c["score"]} for c in context_chunks]
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


def _build_system_prompt(chunks) -> str:
    if not chunks:
        return (
            "You are a helpful assistant. Answer questions clearly and concisely. "
            "If you don't know something, say so."
        )
    context_text = "\n\n---\n\n".join(
        f"[From: {c['filename']}]\n{c['text']}" for c in chunks
    )
    return (
        "You are a helpful assistant with access to the following document excerpts. "
        "Use them to answer the user's question. Always cite which file/document "
        "your information comes from when possible. If the documents don't contain "
        "enough information, use your general knowledge and say so.\n\n"
        "=== DOCUMENT CONTEXT ===\n"
        f"{context_text}\n"
        "========================\n\n"
        "Now answer the user's question based on the above context."
    )


# ── Files ─────────────────────────────────────────────────────────────────────

@app.route("/api/files", methods=["GET"])
def list_files():
    return jsonify(rag_index.list_indexed_files())


@app.route("/api/files/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    results = []
    for file in request.files.getlist("files"):
        filename = file.filename or "unnamed"
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": filename,
                "error": f"File type '{ext}' not supported",
                "ok": False
            })
            continue

        save_path = UPLOAD_DIR / filename
        file.save(str(save_path))
        try:
            meta = rag_index.index_file(str(save_path))
            results.append({**meta, "ok": True})
        except Exception as e:
            results.append({"filename": filename, "error": str(e), "ok": False})

    return jsonify({"results": results})


@app.route("/api/files/<file_hash>", methods=["DELETE"])
def remove_file(file_hash):
    rag_index.remove_file(file_hash)
    return jsonify({"ok": True})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🦙  Ollama Chat is starting...")
    print("    Open  http://localhost:5000  in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
