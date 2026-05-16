import ollama
from tabulate import tabulate
import time
models = [
    "qwen3:1.7b",   # маленькая
    "gemma:latest",     # тож маленькая
]
tasks = [
    {
        "name": "Код",
        "prompt": "Напиши Python-функцию, которая находит все дубликаты в списке. "
                  "Верни только код, без объяснений."
    },
    {
        "name": "Русский текст",
        "prompt": "Объясни в 2 предложениях, чем отличается REST API от FastAPI."
    },
    {
        "name": "Анализ",
        "prompt": "Клиент написал: 'Касса не работает, работа стоит, вы не решаете мою проблему!' "
                  "Определи тональность (positive/negative/neutral) и срочность (high/medium/low)."
    }
]                                   
for model in models:
    print(f"\n{'=' * 60}")
    print(f"Модель: {model}")
    print(f"{'=' * 60}")

    for task in tasks:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}]
        )
        elapsed = time.time() - start
        tokens = response.get("eval_count", 0)
        speed = tokens / elapsed if elapsed > 0 else 0

        # Первые 100 символов ответа
        answer_preview = response["message"]["content"][:100].replace("\n", " ")
        print(f"\n  [{task['name']}]")
        print(f"  Время: {elapsed:.1f}с | Токены: {tokens} | Скорость: {speed:.1f} tok/s")
        print(f"  Ответ: {answer_preview}...")
        
# Пример результата (MacBook M3 16 ГБ): *  Выполнение задачи: C:\Users\m.roschina\AppData\Local\Programs\Python\Python311\python.exe c:\Users\m.roschina\Desktop\Эвотор\мое\Обучение\bag_assistant_project\benchmark_1.py 


#============================================================
#Модель: qwen3:1.7b
#============================================================

#[Код]
# Время: 713.2с | Токены: 566 | Скорость: 0.8 tok/s
#Ответ: def find_duplicates(lst):     counts = {}     for item in lst:         counts[item] = counts.get(ite...

#[Русский текст]
#Время: 217.7с | Токены: 254 | Скорость: 1.2 tok/s
# Ответ: REST API использует HTTP-методы и строится на принципах архитектуры, а FastAPI — это современный фре...

# [Анализ]
# Время: 97.3с | Токены: 378 | Скорость: 3.9 tok/s
# Ответ: **Тональность:** **негативная**   **Срочность:** **средняя**    **Обоснование:**   - **Тональность:*...

#============================================================
#Модель: gemma:latest
#============================================================

#[Код]
#Время: 104.3с | Токены: 36 | Скорость: 0.3 tok/s
#Ответ: ```python def find_duplicates(list1):     return [item for item in list1 if list1.count(item) > 1] `...

#[Русский текст]
#Время: 49.2с | Токены: 60 | Скорость: 1.2 tok/s
#Ответ: REST API использует стандартные RESTful архитектуры, ориентированной на URL-адреса, методы HTTP и ст...

#[Анализ]
#Время: 19.4с | Токены: 14 | Скорость: 0.7 tok/s
#Ответ: **Тональность:** Negative  **Срочность:** High...
#*  Терминал будет повторно использоваться задачами. Чтобы закрыть его, нажмите любую клавишу. 
