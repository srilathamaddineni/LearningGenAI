from openai import OpenAI
from openai import BadRequestError

#Creates an OpenAI client using your API key
client = OpenAI()
print("🤖 GPT Chatbot (Streaming Mode)")
print("Type 'exit' or 'quit' to end the chat.\n")

conversation_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Chatbot: Goodbye! 👋")
        break
    if user_input.lower() == "clear":
        conversation_history.clear()
        print("🧹 Memory cleared! Starting a new conversation.\n")
        continue
    conversation_history.append({"role": "user", "content": user_input})
    chat_input = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_history
    ]
    try:
        # Try to stream the response
        with client.responses.stream(
            model="gpt-4o-mini",
            input=chat_input,
            max_output_tokens=300
        ) as stream:
            print("Chatbot:", end=" ", flush=True)
            full_response = ""
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                    full_response += event.delta
            print("\n")
        conversation_history.append({"role": "assistant", "content": full_response})

    except BadRequestError as e:
        # If streaming isn't allowed, fall back to normal mode
        if "must be verified to stream" in str(e):
            print("\n⚠️ Streaming not supported for your account. Falling back to normal mode.\n")
            response = client.responses.create(
                model="gpt-4o-mini",
                input=chat_input,
                max_output_tokens=300
            )
            assistant_text = response.output_text or ""
            print("Chatbot:", assistant_text, "\n")

            # store assistant reply in memory
            conversation_history.append(
                {"role": "assistant", "content": assistant_text}
            )
        else:
            print(f"❌ Error: {e}")