import os
from typing import List,Tuple
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import json
import hashlib
from pathlib import Path

load_dotenv()  # Load environment variables from .env file


#PDF Text extraction

def extract_pdf_urls(pdf_path: str) -> list[str]:
    """
    Extracts hyperlink URLs stored as PDF annotations (/Annots with /A /URI).
    Returns a de-duplicated list of URLs.
    """
    reader = PdfReader(pdf_path)
    urls = []

    for page in reader.pages:
        annots = page.get("/Annots")
        if not annots:
            continue

        for annot_ref in annots:
            try:
                annot = annot_ref.get_object()
                action = annot.get("/A")
                if action and action.get("/S") == "/URI":
                    uri = action.get("/URI")
                    if uri:
                        urls.append(str(uri))
            except Exception:
                # Some PDFs have weird/unsupported annotation structures
                continue

    # de-duplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

#It generates a unique fingerprint of a file based on its contents.
def file_sha256(path:str)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

# .
def cache_paths(pdf_path:str)->tuple[Path,Path]:
    """
    Stores cache next to your script in a .cache folder.
    One cache per PDF file (based on pdf file hash).
    """
    cache_dir=Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    pdf_hash=file_sha256(pdf_path)[:16]
    base=cache_dir/f"pdf_{pdf_hash}"
    # .json → store text chunks
    # .npy → store embedding vectors
    return base.with_suffix(".json"),base.with_suffix(".npy")

def load_embedding_cache(
        meta_path:Path,
        vecs_path:Path,
        expected_fingerprint:str
)->tuple[list[str],np.ndarray] | None:
    if not meta_path.exists() or not vecs_path.exists():
        return None
    try:
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint")!=expected_fingerprint:
            return None
        chunks=meta["chunks"]
        vecs=np.load(vecs_path)
        return chunks,vecs
    except Exception:
        return None

def save_embedding_cache(
    meta_path: Path,
    vecs_path: Path,
    fingerprint: str,
    chunks: list[str],
    vecs: np.ndarray
) -> None:
    meta = {
        "fingerprint": fingerprint,
        "chunks": chunks,
        "num_chunks": len(chunks),
        "embedding_dim": int(vecs.shape[1]) if vecs.ndim == 2 else None,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(vecs_path, vecs)
    

def extract_pdf_text(pdf_path: str) -> str:
    reader=PdfReader(pdf_path)
    pages_text=[]
    for i, page in enumerate(reader.pages):
        text=page.extract_text() or ""
        text=" ".join(text.split())
        if text:
            pages_text.append(text)
    return "\n".join(pages_text).strip()


  #Chunking
def chunk_text(text:str,max_chars:int=1200,overlap:int=150)->List[str]:
    """
    split text into chunks with a little overlap to keep continuity
    """
    text=" ".join(text.split())
    if not text:
        return []
    chunks=[]
    start=0
    n=len(text)
    while(start<n):
        end=min(start+max_chars,n)
        #try to end near a boundary
        cut=text.rfind(".",start,end)
        if(cut!=-1 and cut>start+int(max_chars*0.6)):
            end=cut+1
        chunk=text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start=max(end-overlap,end)
    return chunks
def embed_texts(client:OpenAI,texts:List[str],embedding_model:str)->np.ndarray:
    """
    Returns a 2D numpy array: shape (len(texts), embedding_dim)
    """
    resp=client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    vectors=[item.embedding for item in resp.data]
    return np.array(vectors,dtype=np.float32)

def cosine_similarity_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    query_vec: (dim,)
    doc_vecs: (N, dim)
    returns: (N,) similarities
    """
    # normalize
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    return d @ q
def top_k_chunks(question: str,
                 chunks: List[str],
                 chunk_vecs: np.ndarray,
                 client: OpenAI,
                 embedding_model: str,
                 k: int = 4) -> List[Tuple[int, float, str]]:
    q_resp = client.embeddings.create(model=embedding_model, input=question)
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)

    sims = cosine_similarity_matrix(q_vec, chunk_vecs)
    top_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[int(i)]) for i in top_idx]
# -------------------------
# Grounded answering (RAG)
# -------------------------

def answer_question_from_chunks(client: OpenAI,
                               question: str,
                               retrieved_chunks: List[Tuple[int, float, str]],
                               llm_model: str) -> str:
    context_blocks = []
    for idx, score, chunk in retrieved_chunks:
        context_blocks.append(f"[CHUNK #{idx} | score={score:.3f}]\n{chunk}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a strict assistant answering questions using ONLY the provided PDF context.

Rules:
- Use ONLY the CONTEXT below.
- If the answer is not explicitly supported by the context, reply exactly:
  I don't know based on the provided document.
- Keep the answer concise (2-6 sentences).
- Do not add outside facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    resp = client.responses.create(
        model=llm_model,
        input=prompt,
        temperature=0.0,
        max_output_tokens=300
    )
    return resp.output_text.strip()
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY (set it in .env or env vars).")

    client = OpenAI(api_key=api_key)

    # You can change these if needed:
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.getenv("LLM_MODEL", "gpt-5.2")

    pdf_path = input("Enter PDF path (e.g., D:\\GenAI\\docs\\file.pdf): ").strip().strip('"')
    if not pdf_path or not os.path.exists(pdf_path):
        raise SystemExit("PDF path is invalid or file does not exist.")

    print("\nReading PDF...")
    text = extract_pdf_text(pdf_path)
    urls = extract_pdf_urls(pdf_path)
    if urls:
        text += "\n\nLINKS FOUND IN PDF:\n" + "\n".join(urls)

    if not text:
        raise SystemExit("Could not extract text from this PDF (it may be scanned images).")

    chunks = chunk_text(text, max_chars=1200, overlap=150)
    meta_path, vecs_path = cache_paths(pdf_path)
    fingerprint = "|".join([
    "v1",  # bump this if you change cache format
    file_sha256(pdf_path),
    embedding_model,
    str(1200),   # max_chars used for chunking (keep in sync)
    str(150),    # overlap used for chunking (keep in sync)
    ])
    cached = load_embedding_cache(meta_path, vecs_path, fingerprint)
    if cached is not None:
        chunks, chunk_vecs = cached
        print(f"Loaded cached embeddings: {len(chunks)} chunks")
    else:
        print("No valid cache found. Creating embeddings...")
        chunk_vecs = embed_texts(client, chunks, embedding_model)
        save_embedding_cache(meta_path, vecs_path, fingerprint, chunks, chunk_vecs)
        print(f"Saved embeddings cache: {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("No chunks created; PDF text seems empty.")

    print(f"PDF loaded. Total chunks: {len(chunks)}")
    print("Creating embeddings (first time can take a bit)...")

    chunk_vecs = embed_texts(client, chunks, embedding_model)

    print("\nAsk questions about the PDF. Type EXIT to quit.\n")

    while True:
        question = input("Q: ").strip()
        if not question:
            continue
        if question.upper() == "EXIT":
            break

        retrieved = top_k_chunks(question, chunks, chunk_vecs, client, embedding_model, k=4)
        answer = answer_question_from_chunks(client, question, retrieved, llm_model)

        print("\nA:", answer)
        print("\nTop chunks used:")
        for idx, score, _ in retrieved:
            print(f"- CHUNK #{idx} (score={score:.3f})")
        print()

if __name__ == "__main__":
    main()