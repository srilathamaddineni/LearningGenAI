import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

def summarize(client: OpenAI, text: str,mode:str) -> str:
    if mode=="bullets":
        instruction=(
            "Summarize the text into 5 bullet points."
            "Each bullet should be under 12 words."
        )
    elif mode=="tldr":
        instruction=(
            "Summarize the text into a 5 bullet  points."
            "Each bullet should be under 12 words."
            )
    else:
        instruction=(
            "Return two sections:\n"
            "1) A one-sentence summary of the text.\n"
            " 2) A list of 3 key takeaways from the text."
            "Follow the format:\n"
        )
    prompt=f"""
{instruction}
Text:
{text}
""".strip()
    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        temperature=0.8,
        max_output_tokens=500
    )
    return response.output_text.strip()

def answer_from_context(client: OpenAI, context: str, question: str) -> str:
    prompt = f"""
You are a strict question-answering assistant.

Rules:
- Use ONLY the provided CONTEXT.
- If the answer is not explicitly in the context, reply exactly:
  I don't know based on the provided context.
- Keep the answer short (1-3 sentences).
- Do not add outside facts.
CONTEXT:
{context}
QUESTION:
{question}
ANSWER:
""".strip()
    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        temperature=0.0,
        max_output_tokens=200
    )
    return response.output_text.strip()

def generate_resume_bullets(client, role: str, experience_text: str) -> str:
    prompt = f"""
You are a senior technical recruiter.

Task:
Rewrite the EXPERIENCE into strong resume bullets for the ROLE.

Rules:
- Output exactly 5 bullets.
- Each bullet must start with a strong action verb (Built, Designed, Improved, Automated, etc.).
- Each bullet must be 1 line and under 22 words.
- Use past tense.
-Focus on the grammar and impact of the bullet, not just the task.
- Do NOT invent tools, numbers, or results. If metrics are missing, keep it general.
- Focus on engineering impact: performance, reliability, automation, scalability, cost, security, teamwork.

ROLE:
{role}

EXPERIENCE:
{experience_text}

OUTPUT (exactly 5 bullets):
""".strip()

    resp = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        temperature=0.2,
        max_output_tokens=250,
    )
    return resp.output_text.strip()


def read_multiline_input() -> str:
    print("Paste your text. Type END and press Enter.")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    print("\nChoose a project:")
    print("1) Project 1 – Text Summarizer")
    print("2) Project 2 – Context-based Q&A")
    print("3) Project 3 – Resume Bullet Generator")

    project_choice = input("Enter 1 or 2 or 3: ").strip()

    # -------------------------
    # Project 1: Text Summarizer
    # -------------------------
    if project_choice == "1":
        print("\nText Summarizer")
        print("Choose output:")
        print("1) Bullet summary")
        print("2) One-line TL;DR")
        print("3) Both")

        choice = input("Enter 1/2/3: ").strip()
        mode_map = {"1": "bullets", "2": "tldr", "3": "both"}
        mode = mode_map.get(choice)

        if not mode:
            raise SystemExit("Invalid summarizer choice.")

        text = read_multiline_input()
        if not text:
            raise SystemExit("No text provided.")

        result = summarize(client, text, mode)
        print("\n--- RESULT ---")
        print(result)

    # -------------------------
    # Project 2: Context-based Q&A
    # -------------------------
    elif project_choice == "2":
        print("\nContext-based Q&A")

        context = read_multiline_input()
        if not context:
            raise SystemExit("Context is empty.")

        question = input("\nEnter your QUESTION: ").strip()
        if not question:
            raise SystemExit("Question is empty.")

        answer = answer_from_context(client, context, question)
        print("\n--- ANSWER ---")
        print(answer)
    elif project_choice == "3":
        print("\nResume Bullet Generator")

        role = input("Enter the ROLE (e.g., Java Backend Engineer): ").strip()
        if not role:
            raise SystemExit("Role is empty.")

        print("\nEnter your EXPERIENCE description. Type END and press Enter.")
        experience_text = read_multiline_input()
        if not experience_text:
            raise SystemExit("Experience description is empty.")

        bullets = generate_resume_bullets(client, role, experience_text)
        print("\n--- RESUME BULLETS ---")
        print(bullets)

    else:
        raise SystemExit("Invalid project choice.")

if __name__ == "__main__":
    main()