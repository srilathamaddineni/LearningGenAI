import os
from typing import List,Tuple
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
load_dotenv()  # Load environment variables from .env file

#PDF Text extraction

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
    if not text:
        raise SystemExit("Could not extract text from this PDF (it may be scanned images).")

    chunks = chunk_text(text, max_chars=1200, overlap=150)
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