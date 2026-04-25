from fastapi import FastAPI, UploadFile, File, Form
import pypdf
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Any, Optional

# Optional cross-encoder for reranking (graceful fallback if not installed)
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

app = FastAPI()

# Configurable parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CANDIDATES = 20
FINAL_TOP_K = 3
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_THRESHOLD = 0.0  # adjust >0.0 to require a minimum rerank score
VECTOR_SIM_THRESHOLD = 0.0

# 1. Initialize the embedding model (converts text to math/vectors)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to the Chroma database directory
# Declared globally so all endpoints can access the same memory
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Try to initialize a CrossEncoder reranker (small model). If unavailable, RERANKER stays None.
RERANKER: Optional[CrossEncoder] = None
if CrossEncoder is not None:
    try:
        RERANKER = CrossEncoder(RERANKER_MODEL)
    except Exception:
        RERANKER = None


# Helper: simple deterministic chunking that returns per-chunk metadata
def chunk_text(text: str, page_number: Optional[int] = None, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split `text` into overlapping chunks and return list of {"text": ..., "metadata": {...}}.
    Metadata contains page_number, chunk_index, start_char, end_char.
    """
    chunks: List[Dict[str, Any]] = []
    if not text:
        return chunks
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    text_len = len(text)
    idx = 0
    chunk_index = 0
    while idx < text_len:
        start = idx
        end = min(idx + chunk_size, text_len)
        piece = text[start:end]
        metadata = {
            "page_number": page_number,
            "chunk_index": chunk_index,
            "start_char": start,
            "end_char": end,
        }
        chunks.append({"text": piece, "metadata": metadata})
        chunk_index += 1
        idx += step
    return chunks


@app.get("/")
def read_root():
    return {"message": "RAG API is running!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Read the file from the incoming request into memory
    contents = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))

    # Loop through all pages and extract text
    # We'll build page-aware chunks with metadata
    all_texts: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    global_chunk_counter = 0

    for page_number, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if not text:
            continue
        # Split this page deterministically into chunks with offsets
        page_chunks = chunk_text(text, page_number=page_number, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for pc in page_chunks:
            all_texts.append(pc["text"])
            md = pc["metadata"].copy()
            md.update({
                "source_file": file.filename,
                "chunk_number": global_chunk_counter + 1,
            })
            all_metadatas.append(md)
            global_chunk_counter += 1

    # If no pages/chunks were found, return a helpful message
    if not all_texts:
        return {"filename": file.filename, "status": "No text extracted", "chunks_added": 0}

    # Save both the text chunks and their corresponding metadata into ChromaDB
    vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas)

    return {
        "filename": file.filename,
        "status": "Success",
        "chunks_added": len(all_texts)
    }


@app.post("/search")
async def search_document(question: str = Form(...)):
    # Perform a similarity search in the vector database and retrieve distances
    # Use a larger candidate set, we'll rerank and trim to FINAL_TOP_K
    try:
        results_with_scores = vectorstore.similarity_search_with_score(question, k=TOP_K_CANDIDATES)
    except Exception:
        # Fallback if the method isn't available on this vectorstore
        simple_results = vectorstore.similarity_search(question, k=FINAL_TOP_K)
        formatted_results = []
        for i, doc in enumerate(simple_results):
            chunk_num = doc.metadata.get("chunk_number", "Unknown")
            source = doc.metadata.get("source_file", "Unknown")
            formatted_results.append({
                "rank": i + 1,
                "chunk_number": chunk_num,
                "source_file": source,
                "content": doc.page_content
            })
        return {"question": question, "top_matches": formatted_results}

    # results_with_scores is a list of (Document, distance) tuples
    candidates_texts: List[str] = []
    candidates_metadatas: List[Dict[str, Any]] = []
    vector_distances: List[float] = []

    for item in results_with_scores:
        # item may be (doc, score) or similar; handle both
        try:
            doc, dist = item
        except Exception:
            # If the response shape is unexpected, skip
            continue
        candidates_texts.append(doc.page_content)
        candidates_metadatas.append(getattr(doc, "metadata", {}) or {})
        vector_distances.append(dist if dist is not None else 0.0)

    # Convert distances to a simple similarity score (distance metric dependent)
    vector_similarities = [1.0 / (1.0 + float(d)) if d is not None else 0.0 for d in vector_distances]

    final_items: List[Dict[str, Any]] = []

    if RERANKER is not None and len(candidates_texts) > 0: # <------------------------- RERANKING LOGIC, README FROM HERE
        # Prepare pairs and call the cross-encoder for reranking
        pairs = [(question, t) for t in candidates_texts]
        try:
            rerank_scores = RERANKER.predict(pairs, batch_size=16)
        except Exception:
            rerank_scores = None

        if rerank_scores is not None:
            combined = []
            for i, text in enumerate(candidates_texts):
                combined.append({
                    "text": text,
                    "metadata": candidates_metadatas[i],
                    "vector_score": vector_similarities[i],
                    "vector_distance": vector_distances[i],
                    "rerank_score": float(rerank_scores[i]),
                })
            # Sort by rerank score desc, dedupe exact texts, apply thresholds
            seen_texts = set()
            kept = []
            for item in sorted(combined, key=lambda x: x["rerank_score"], reverse=True):
                t = item["text"].strip()
                if t in seen_texts:
                    continue
                seen_texts.add(t)
                if item["rerank_score"] < RERANK_THRESHOLD:
                    continue
                if item["vector_score"] < VECTOR_SIM_THRESHOLD:
                    continue
                kept.append(item)
                if len(kept) >= FINAL_TOP_K:
                    break

            # Format for return to match previous shape but with more info
            for rank_idx, it in enumerate(kept):
                md = it.get("metadata", {})
                chunk_num = md.get("chunk_number", "Unknown")
                source = md.get("source_file", "Unknown")
                final_items.append({
                    "rank": rank_idx + 1,
                    "chunk_number": chunk_num,
                    "source_file": source,
                    "content": it.get("text"),
                    "vector_score": it.get("vector_score"),
                    "rerank_score": it.get("rerank_score"),
                })

            return {"question": question, "top_matches": final_items}

    # Fallback: return top by vector similarity
    for i in range(min(FINAL_TOP_K, len(candidates_texts))):
        md = candidates_metadatas[i]
        chunk_num = md.get("chunk_number", "Unknown")
        source = md.get("source_file", "Unknown")
        final_items.append({
            "rank": i + 1,
            "chunk_number": chunk_num,
            "source_file": source,
            "content": candidates_texts[i],
            "vector_score": vector_similarities[i],
            "vector_distance": vector_distances[i],
        })

    return {"question": question, "top_matches": final_items}
