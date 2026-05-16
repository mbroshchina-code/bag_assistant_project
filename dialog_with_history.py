import ollama
from ollama import chat

def chatbot(model: str = "bag_assistant_model:latest"):
    """Простой чат-бот с историей диалога."""
    print(f"Чат-бот на модели {model}. Введите 'выход' для завершения.\n")

    # История диалога — передаётся в каждом запросе
    messages = [
        {
            "role": "system",
            "content": "Ты — дружелюбный ассистент техподдержки "
                       "Отвечай на русском, вежливо, уважительно. Старайся оносить информацию понятно, кратко, но без потери ключевых моментов."
        }
    ]

    while True:
        user_input = input("Вы: ").strip()
        if user_input.lower() in ("выход", "exit", "quit"):
            print("До свидания!")
            break
        if not user_input:
            continue

        # Добавляем сообщение пользователя в историю
        messages.append({"role": "user", "content": user_input})

        # Отправляем ВСЮ историю модели (стриминг для UX)
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

        print()  # перенос строки после ответа

        # Добавляем ответ модели в историю
        messages.append({"role": "assistant", "content": full_response})

        # Ограничиваем историю (чтобы не превысить контекстное окно)
        if len(messages) > 21:  # system + 10 пар user/assistant
            messages = [messages[0]] + messages[-20:]  # сохраняем system + последние 20
            print("[Историю обрезали до последних 10 сообщений]")
            
chatbot()