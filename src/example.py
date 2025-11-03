from openai import OpenAI
from openai import BadRequestError

#Creates an OpenAI client using your API key
client = OpenAI()
print("🤖 GPT Chatbot (Streaming Mode)")
print("Type 'exit' or 'quit' to end the chat.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Chatbot: Goodbye! 👋")
        break

    try:
        # Try to stream the response
        with client.responses.stream(
            model="gpt-5",
            input=user_input,
            max_output_tokens=300
        ) as stream:
            print("Chatbot:", end=" ", flush=True)
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
            print("\n")

    except BadRequestError as e:
        # If streaming isn't allowed, fall back to normal mode
        if "must be verified to stream" in str(e):
            print("\n⚠️ Streaming not supported for your account. Falling back to normal mode.\n")
            response = client.responses.create(
                model="gpt-5",
                input=user_input,
                max_output_tokens=300
            )
            print("Chatbot:", response.output_text, "\n")
        else:
            print(f"❌ Error: {e}")