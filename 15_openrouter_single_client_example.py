import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
USE_LOCAL = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if USE_LOCAL:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    model = "qwen3:8b"
else:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    model = "gpt-5-mini"
from openai import OpenAI
import os
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# список моделей (fallback цепочка)
MODELS = [
    "openai/gpt-oss-120b:free",
    # "tencent/hy3-preview:free",
]

def ask_model(prompt, max_retries=2):
    for model in MODELS:
        for attempt in range(max_retries):
            try:
                print(f"Trying {model} (attempt {attempt+1})")

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"Error with {model}: {e}")
                time.sleep(2 ** attempt)  # exponential backoff

        print(f"Switching model...\n")

    raise Exception("Все модели недоступны 😢")


# вызов
answer = ask_model("Ты - чат-бот техподдержки. Тебе нужно определить категорию проблемы из сообщения пользователя. Доступные категории: [Проблема с оплатой], [Ошибка авторизации], [Аппартная неисправность], [Проблема с интернетом]. Проанализируй текст и выведи ТОЛЬКО название категории в квадратных скобках, без лишних слов. Текст пользователя: 'хелп у меня терминал ваще нне алё сдох экран негорит и пикает скажите срочно че делать'.")
print(answer)

#Результат
#*  Выполнение задачи: C:\Users\m.roschina\AppData\Local\Programs\Python\Python311\python.exe c:\Users\m.roschina\Desktop\Эвотор\мое\Обучение\bag_assistant_project\15_openrouter_single_client_example.py 

#Trying openai/gpt-oss-120b:free (attempt 1)
#Привет! Как могу помочь?
# *  Терминал будет повторно использоваться задачами. Чтобы закрыть его, нажмите любую клавишу. 
#
