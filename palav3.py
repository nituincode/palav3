import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np

import faiss  # faiss-cpu
from openai import OpenAI

# Optional: YouTube transcript support (recommended)
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

# Optional: PDF text extraction
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# ----------------------------
# Config
# ----------------------------

DEFAULT_LINKS_FILE = "palav_url_links.txt"  # put this in your repo (same folder as app.py)

# Retrieval + chunking knobs
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 250
TOP_K = 6
MIN_SIM_THRESHOLD = 0.30  # raise => more strict "not found"; lower => more answers

# Embeddings model used for indexing + querying
EMBED_MODEL = "text-embedding-3-small"  # can switch to text-embedding-3-large

# Answer model (you can try gpt-5.2 after everything works)
ANSWER_MODEL_DEFAULT = "gpt-4.1-mini"

# Persistence directory
INDEX_DIR = ".palav_index_cache"  # local folder in app working directory


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class DocChunk:
    id: str
    source_url: str
    title: str
    text: str


# ----------------------------
# Utilities
# ----------------------------

def normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")

def is_youtube_url(url: str) -> bool:
    u = url.lower()
    return ("youtube.com/watch" in u) or ("youtu.be/" in u)

def extract_youtube_video_id(url: str) -> Optional[str]:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    m = re.search(r"[?&]v=([^&]+)", url)
    if m:
        return m.group(1)
    return None


# ----------------------------
# Fetch + Extract
# ----------------------------

def fetch_html_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url
    main = soup.find("main") or soup.find("article")
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    return title, normalize_whitespace(text)

def fetch_pdf_text(url: str, timeout: int = 30) -> Tuple[str, str]:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Add pypdf to requirements.txt")

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    from io import BytesIO
    reader = PdfReader(BytesIO(r.content))

    title = url
    try:
        meta = reader.metadata
        if meta and getattr(meta, "title", None):
            title = meta.title
    except Exception:
        pass

    pages_text = []
    for p in reader.pages:
        try:
            pages_text.append(p.extract_text() or "")
        except Exception:
            pages_text.append("")

    return title, normalize_whitespace("\n".join(pages_text))

def fetch_youtube_transcript_text(url: str) -> Tuple[str, str]:
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api is not installed. Add it to requirements.txt")

    vid = extract_youtube_video_id(url)
    if not vid:
        raise RuntimeError("Could not parse YouTube video id")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
    except Exception:
        transcript = YouTubeTranscriptApi.get_transcript(vid)

    text = " ".join([x.get("text", "") for x in transcript])
    return f"YouTube transcript: {vid}", normalize_whitespace(text)


# ----------------------------
# Chunking
# ----------------------------

def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ----------------------------
# OpenAI Embeddings + FAISS
# ----------------------------

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs

def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim after normalization
    index.add(vectors)
    return index


# ----------------------------
# Load URLs from file
# ----------------------------

def load_allowed_urls(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            found = re.findall(r"https?://\S+", line)
            urls.extend(found if found else [line])

    seen = set()
    out = []
    for u in urls:
        u = u.strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ----------------------------
# Persistence helpers
# ----------------------------

def ensure_index_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)

def index_key(links_file: str) -> str:
    """
    Key changes when links file content changes, or when you change EMBED_MODEL / chunking knobs.
    """
    if not os.path.exists(links_file):
        return "missing_links_file"
    h = file_sha1(links_file)
    settings = f"{EMBED_MODEL}|{CHUNK_CHARS}|{CHUNK_OVERLAP}"
    return sha1(h + "|" + settings)

def index_paths(key: str) -> Dict[str, str]:
    return {
        "faiss": os.path.join(INDEX_DIR, f"{key}.faiss"),
        "vectors": os.path.join(INDEX_DIR, f"{key}.npy"),
        "chunks": os.path.join(INDEX_DIR, f"{key}.chunks.jsonl"),
        "report": os.path.join(INDEX_DIR, f"{key}.report.json"),
        "meta": os.path.join(INDEX_DIR, f"{key}.meta.json"),
    }

def index_exists(paths: Dict[str, str]) -> bool:
    return all(os.path.exists(paths[p]) for p in ["faiss", "vectors", "chunks", "meta"])

def save_index(paths: Dict[str, str], index, vectors: np.ndarray, chunks: List[DocChunk], report: Dict):
    faiss.write_index(index, paths["faiss"])
    np.save(paths["vectors"], vectors)

    # Save chunks as JSONL (safer than pickle)
    with open(paths["chunks"], "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "id": c.id,
                "source_url": c.source_url,
                "title": c.title,
                "text": c.text,
            }, ensure_ascii=False) + "\n")

    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump({
            "embed_model": EMBED_MODEL,
            "chunk_chars": CHUNK_CHARS,
            "chunk_overlap": CHUNK_OVERLAP,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

def load_index(paths: Dict[str, str]) -> Tuple[object, np.ndarray, List[DocChunk], Dict]:
    index = faiss.read_index(paths["faiss"])
    vectors = np.load(paths["vectors"])

    chunks: List[DocChunk] = []
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(DocChunk(**obj))

    report = {}
    if os.path.exists(paths["report"]):
        with open(paths["report"], "r", encoding="utf-8") as f:
            report = json.load(f)

    return index, vectors, chunks, report


# ----------------------------
# Ingest + Build (one-time)
# ----------------------------

def ingest_sources(links_file: str) -> Tuple[List[DocChunk], Dict]:
    urls = load_allowed_urls(links_file)
    report = {"total_urls": len(urls), "ok": 0, "failed": []}

    chunks: List[DocChunk] = []
    for url in urls:
        try:
            if is_youtube_url(url):
                title, text = fetch_youtube_transcript_text(url)
            elif is_pdf_url(url):
                title, text = fetch_pdf_text(url)
            else:
                title, text = fetch_html_text(url)

            if len(text) < 200:
                raise RuntimeError("Extracted text too short (blocked or mostly non-text).")

            for i, piece in enumerate(chunk_text(text)):
                chunk_id = sha1(url + f"::{i}::{piece[:80]}")
                chunks.append(DocChunk(
                    id=chunk_id,
                    source_url=url,
                    title=title,
                    text=piece
                ))

            report["ok"] += 1
        except Exception as e:
            report["failed"].append({"url": url, "error": repr(e)})

        time.sleep(0.2)

    return chunks, report

def build_or_load(links_file: str, api_key: str, force_rebuild: bool = False):
    """
    Loads persisted index if available; otherwise builds, saves, and returns it.
    """
    ensure_index_dir()
    key = index_key(links_file)
    paths = index_paths(key)

    if (not force_rebuild) and index_exists(paths):
        index, vectors, chunks, report = load_index(paths)
        return index, vectors, chunks, report, key, paths, True

    # Build fresh
    chunks, report = ingest_sources(links_file)
    if not chunks:
        raise RuntimeError("No content chunks were created. Check links reachability or extraction errors.")

    client = OpenAI(api_key=api_key)
    texts = [c.text for c in chunks]
    vectors = embed_texts(client, texts)
    index = build_faiss_index(vectors)

    save_index(paths, index, vectors, chunks, report)
    return index, vectors, chunks, report, key, paths, False


# ----------------------------
# Retrieval + Answer
# ----------------------------

SYSTEM_INSTRUCTIONS = """You are a breastfeeding education chatbot for an NGO.
You MUST answer using only the provided SOURCES (snippets).
If the SOURCES do not contain the answer, reply exactly:
"I do not have required information. Please try different question"
Keep the tone parent-friendly and practical.
Do not provide medical diagnosis. Encourage contacting a lactation consultant/doctor for urgent issues.
Always include citations as a bulleted list of the source URLs you used at the end under 'Sources:'.
"""

def retrieve(client: OpenAI, index, chunks: List[DocChunk], query: str, top_k: int = TOP_K):
    qvec = embed_texts(client, [query])
    sims, idxs = index.search(qvec, top_k)
    sims = sims[0].tolist()
    idxs = idxs[0].tolist()

    results = []
    for score, i in zip(sims, idxs):
        if i == -1:
            continue
        results.append((float(score), chunks[i]))
    return results

def make_answer(client: OpenAI, model: str, question: str, retrieved: List[Tuple[float, DocChunk]]) -> str:
    best = retrieved[0][0] if retrieved else 0.0
    if best < MIN_SIM_THRESHOLD:
        return "I do not have required information. Please try different question."

    context_blocks = []
    used_urls = []
    for score, ch in retrieved[:TOP_K]:
        context_blocks.append(
            f"URL: {ch.source_url}\nTITLE: {ch.title}\nSNIPPET:\n{ch.text}\n"
        )
        used_urls.append(ch.source_url)

    used_urls_unique = []
    seen = set()
    for u in used_urls:
        if u not in seen:
            seen.add(u)
            used_urls_unique.append(u)

    user_msg = f"""QUESTION:
{question}

SOURCES:
{chr(10).join(context_blocks)}

Remember: If the SOURCES do not contain the answer, say exactly:
"I do not have required information. Please try different question."
"""

    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[{"role": "user", "content": user_msg}],
    )
    answer = (getattr(resp, "output_text", "") or "").strip()

    if "Sources:" not in answer:
        answer = answer.rstrip() + "\n\nSources:\n" + "\n".join([f"- {u}" for u in used_urls_unique])

    return answer


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Palav Breastfeeding Chatbot", layout="centered")
st.title("Palav Breastfeeding Userguide")

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not api_key:
    st.error("OPENAI_API_KEY is not set. Add it to Streamlit secrets or environment variables.")
    st.stop()
# Read admin flag (define ADMIN_MODE properly)
ADMIN_MODE = str(
    st.secrets.get("ADMIN_MODE", os.getenv("ADMIN_MODE", "false"))
).lower() in {"1", "true", "yes"}

# (Optional) debug - use Streamlit, not print
st.write("ADMIN_MODE:", ADMIN_MODE)

# Defaults for normal users
links_file = DEFAULT_LINKS_FILE
answer_model = ANSWER_MODEL_DEFAULT
force_rebuild = False

# Admin-only controls (proper indentation!)
if ADMIN_MODE:
    with st.expander("Admin (optional)", expanded=False):
        links_file = st.text_input("Links file path", value=DEFAULT_LINKS_FILE)
        answer_model = st.text_input("Answer model", value=ANSWER_MODEL_DEFAULT)
        force_rebuild = st.checkbox("Force rebuild index now", value=False)

# Build or load index (persisted)
try:
    with st.spinner("Loading index (or building it once if missing)..."):
        index, vectors, chunks, report, key, paths, loaded_from_disk = build_or_load(
            links_file=links_file,
            api_key=api_key,
            force_rebuild=force_rebuild,
        )
except Exception as e:
    st.error(f"Index load/build failed: {repr(e)}")
    st.stop()

# Status panel
if ADMIN_MODE:
    with st.expander("Index status", expanded=False):
        st.write(f"Index key: `{key}`")
        st.write("Loaded from disk:", loaded_from_disk)
        st.write("Stored files:")
        st.code("\n".join([f"{k}: {v}" for k, v in paths.items()]))
        st.write("Ingest report (last build):")
        st.json(report if report else {"note": "No report found (older cache)."})

#st.info(
#   "Note: On Streamlit Cloud, saved files persist only while the container stays alive. "
#   "If the app sleeps/restarts or you redeploy, it may rebuild. "
#   "For permanent persistence, store the index in S3/Drive/DB."
# )

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Palav Breastifeeding Userguide. Ask me any breastfeeding question. I will answer only from the approved links."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Type your question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    client = OpenAI(api_key=api_key)

    with st.chat_message("assistant"):
        with st.spinner("Searching approved sources..."):
            retrieved = retrieve(client, index, chunks, question, top_k=TOP_K)
            answer = make_answer(client, answer_model, question, retrieved)
            st.markdown(answer)

            with st.expander("Debug (retrieval scores)", expanded=False):
                st.write("Top matches:")
                for score, ch in retrieved[:TOP_K]:
                    st.write({"score": round(score, 4), "url": ch.source_url, "title": ch.title})

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
