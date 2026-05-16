import ollama

from ollama import chat

response = chat(
    model='qwen3:1.7b',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)

def chatbot(model: str = "qwen3:1.7b"):
    print(f"Чат-бот на модели {model}. Введите 'выход' для завершения.\n")
    messages = [
        {
            "role": "system",
            "content": "Ты — дружелюбный помощник разработчика. "
                       "Отвечай на русском, кратко и с примерами кода, когда уместно."
        }
    ]

    while True:
        user_input = input("Вы: ").strip()
        if user_input.lower() in ("выход", "exit", "quit"):
            print("До свидания!")
            break
        if not user_input:
            continue 
        messages.append({"role": "user", "content": user_input})
        print("Бот: ", end="")
        full_response = ""

        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token

        print()
        messages.append({"role": "assistant", "content": full_response})
        if len(messages) > 21:  # system + 10 пар user/assistant
            messages = [messages[0]] + messages[-20:]  # сохраняем system + последние 20
            print("[Историю обрезали до последних 10 сообщений]")
chatbot()