import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

load_dotenv()


def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    patterns = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    raise ValueError("Could not extract video id. Paste a full YouTube URL or the 11-char video id.")


def fetch_transcript(video_id: str, languages=None) -> str:
    languages = languages or ["en"]

    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=languages)  # <-- NEW API

    
    return " ".join(snippet.text.replace("\n", " ").strip() for snippet in fetched).strip()


def chunk_text(text: str, max_chars: int = 6500) -> List[str]:
    """
    Simple char-based chunking with a soft sentence boundary.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to cut near a period
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.6):
            cut = end
        else:
            cut += 1
        chunks.append(text[start:cut].strip())
        start = cut

    return [c for c in chunks if c]


def summarize_chunk(client: OpenAI, chunk: str, i: int, total: int) -> str:
    prompt = f"""
You are summarizing a YouTube transcript chunk.

Return EXACTLY this format:

CHUNK_SUMMARY:
<2-3 sentences>

KEY_POINTS:
- <bullet 1>
- <bullet 2>
- <bullet 3>

Chunk {i}/{total}:
{chunk}
""".strip()

    resp = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        temperature=0.2,
        max_output_tokens=350,
    )
    return resp.output_text.strip()


def combine_summaries(client: OpenAI, chunk_summaries: List[str]) -> str:
    joined = "\n\n---\n\n".join(chunk_summaries)

    prompt = f"""
You are combining partial summaries into one final summary.

Rules:
- Do NOT invent facts.
- Prefer clarity over length.
- Output EXACTLY this format:

TITLE:
<best guess title>

TLDR:
<one line>

SUMMARY:
<5-8 bullets>

KEY TAKEAWAYS:
- <3-6 bullets>

CHAPTERS (optional):
- <timestamp-like label if you can infer, else general section names>

CHUNK SUMMARIES:
{joined}
""".strip()

    resp = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        temperature=0.2,
        max_output_tokens=700,
    )
    return resp.output_text.strip()


# -------------------------
# Main pipeline
# -------------------------

def summarize_youtube(url_or_id: str, languages: Optional[List[str]] = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY (set it in .env or env vars).")

    client = OpenAI(api_key=api_key)

    video_id = extract_video_id(url_or_id)
    transcript = fetch_transcript(video_id, languages=languages)

    chunks = chunk_text(transcript, max_chars=6500)
    print(f"\nTranscript length: {len(transcript)} chars | Chunks: {len(chunks)}\n")

    chunk_summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {idx}/{len(chunks)}...")
        chunk_summaries.append(summarize_chunk(client, chunk, idx, len(chunks)))

    print("\nCombining into final summary...\n")
    return combine_summaries(client, chunk_summaries)


def main():
    url = input("Paste YouTube URL or Video ID: ").strip()
    if not url:
        raise SystemExit("No URL or ID provided.")

    # You can change languages=["en"] to ["en", "hi"] etc.
    result = summarize_youtube(url, languages=["en"])
    print("\n--- FINAL SUMMARY ---\n")
    print(result)
    print()

if __name__ == "__main__":
    main()