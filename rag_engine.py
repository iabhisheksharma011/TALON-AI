"""
RAG Engine — pure stdlib + pypdf, no external vector DB.
Files are chunked, TF-IDF vectors computed, and cached to disk
so repeat loads are near-instant.
"""

import os
import re
import json
import math
import hashlib
import sqlite3
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
DB_PATH = CACHE_DIR / "rag.db"

CHUNK_SIZE = 600        # chars per chunk
CHUNK_OVERLAP = 100     # overlap between chunks
TOP_K = 5              # retrieved chunks per query


# ── Database ────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_chunks (
            file_hash TEXT NOT NULL,
            filename  TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            PRIMARY KEY (file_hash, chunk_idx)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_meta (
            file_hash TEXT PRIMARY KEY,
            filename  TEXT NOT NULL,
            size_bytes INTEGER,
            chunk_count INTEGER,
            indexed_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf not installed")
        reader = PdfReader(filepath)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    elif ext in (".txt", ".md", ".csv", ".json", ".py", ".js", ".ts",
                 ".html", ".htm", ".xml", ".yaml", ".yml", ".rst", ".log"):
        with open(filepath, "r", errors="replace") as f:
            return f.read()
    else:
        # Try reading as text anyway
        try:
            with open(filepath, "r", errors="replace") as f:
                return f.read()
        except Exception:
            raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 30]


# ── TF-IDF retrieval ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


def _build_idf(chunks: List[str]) -> Dict[str, float]:
    N = len(chunks)
    df: Dict[str, int] = defaultdict(int)
    for chunk in chunks:
        for tok in set(_tokenize(chunk)):
            df[tok] += 1
    return {t: math.log((N + 1) / (d + 1)) + 1 for t, d in df.items()}


def _tfidf_vec(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = _tf(tokens)
    return {t: tf[t] * idf.get(t, 1.0) for t in tf}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── In-memory index (per session) ────────────────────────────────────────────

class RAGIndex:
    """Holds all chunks + TF-IDF index across all loaded files."""

    def __init__(self):
        self._chunks: List[Tuple[str, str, str]] = []  # (file_hash, filename, text)
        self._vecs: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._loaded_hashes: set = set()
        self._db = _get_db()

    # ── Public API ───────────────────────────────────────────────────────────

    def index_file(self, filepath: str) -> Dict:
        """Index a file. Returns metadata dict. Cached if already seen."""
        fhash = _file_hash(filepath)
        filename = Path(filepath).name

        already = self._db.execute(
            "SELECT chunk_count FROM file_meta WHERE file_hash=?", (fhash,)
        ).fetchone()

        if already:
            chunk_count = already[0]
            cached = True
            if fhash not in self._loaded_hashes:
                self._load_from_db(fhash)
        else:
            text = extract_text(filepath)
            chunks = _chunk_text(text)
            chunk_count = len(chunks)
            self._save_to_db(fhash, filename, filepath, chunks)
            self._load_from_db(fhash)
            cached = False

        self._rebuild_idf()
        return {
            "file_hash": fhash,
            "filename": filename,
            "chunk_count": chunk_count,
            "cached": cached,
        }

    def list_indexed_files(self) -> List[Dict]:
        rows = self._db.execute(
            "SELECT file_hash, filename, size_bytes, chunk_count, indexed_at FROM file_meta"
        ).fetchall()
        return [
            {"file_hash": r[0], "filename": r[1], "size": r[2],
             "chunks": r[3], "indexed_at": r[4]}
            for r in rows
        ]

    def remove_file(self, file_hash: str):
        self._db.execute("DELETE FROM file_chunks WHERE file_hash=?", (file_hash,))
        self._db.execute("DELETE FROM file_meta WHERE file_hash=?", (file_hash,))
        self._db.commit()
        self._chunks = [(h, fn, t) for h, fn, t in self._chunks if h != file_hash]
        self._loaded_hashes.discard(file_hash)
        self._rebuild_idf()

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if not self._chunks:
            return []
        q_tokens = _tokenize(query)
        q_vec = _tfidf_vec(q_tokens, self._idf)
        scores = [
            (_cosine(q_vec, v), i) for i, v in enumerate(self._vecs)
        ]
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score < 0.01:
                continue
            fhash, fname, text = self._chunks[idx]
            results.append({"filename": fname, "text": text, "score": round(score, 4)})
        return results

    # ── Internal ─────────────────────────────────────────────────────────────

    def _save_to_db(self, fhash, filename, filepath, chunks):
        size = os.path.getsize(filepath)
        self._db.execute(
            "INSERT OR REPLACE INTO file_meta (file_hash, filename, size_bytes, chunk_count) VALUES (?,?,?,?)",
            (fhash, filename, size, len(chunks))
        )
        self._db.executemany(
            "INSERT OR REPLACE INTO file_chunks (file_hash, filename, chunk_idx, chunk_text) VALUES (?,?,?,?)",
            [(fhash, filename, i, c) for i, c in enumerate(chunks)]
        )
        self._db.commit()

    def _load_from_db(self, fhash):
        rows = self._db.execute(
            "SELECT filename, chunk_text FROM file_chunks WHERE file_hash=? ORDER BY chunk_idx",
            (fhash,)
        ).fetchall()
        for fname, text in rows:
            self._chunks.append((fhash, fname, text))
        self._loaded_hashes.add(fhash)

    def _rebuild_idf(self):
        texts = [t for _, _, t in self._chunks]
        self._idf = _build_idf(texts)
        self._vecs = [_tfidf_vec(_tokenize(t), self._idf) for t in texts]

    def load_all_from_db(self):
        """Restore all previously indexed files into memory on startup."""
        hashes = [r[0] for r in self._db.execute("SELECT file_hash FROM file_meta").fetchall()]
        for fhash in hashes:
            if fhash not in self._loaded_hashes:
                self._load_from_db(fhash)
        if hashes:
            self._rebuild_idf()


# Singleton
rag_index = RAGIndex()
rag_index.load_all_from_db()
