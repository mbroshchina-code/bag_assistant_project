import os
from openai import OpenAI
import ollama
from dotenv import load_dotenv
from real_eval_tasks import EVAL_TASKS
import time

load_dotenv()
print("Существует ли файл .env?:", os.path.exists(".env"))

# Настраиваем клиент для облачных моделей OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# СПИСОК ВСЕХ МОДЕЛЕЙ ДЛЯ СРАВНЕНИЯ
# Чтобы код понимал, где какая модель, добавим к ним префиксы 'ollama/' и 'openrouter/'
MODELS_TO_TEST = [
    "openrouter/openai/gpt-oss-120b:free", # Облачная бесплатная
    "openrouter/google/gemma-4-26b-a4b-it:free",              # Облачная быстрая
]

def get_generation(model_tag: str, prompt: str) -> str:
    """Универсальный генератор ответа в зависимости от типа модели"""
    # Разделяем наш тег на тип источника и реальное имя модели
    source, model_name = model_tag.split("/", 1)
    
    if source == "ollama":
        # Запрос к локальной Ollama
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0} # Фиксируем креативность для EVAL
        )
        return response.choices[0].message.content
    
    elif source == "openrouter":
        # Запрос к облачному OpenRouter
        MAX_RETRIES = 5
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Trying {model_name} (attempt {attempt+1})")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                time.sleep(2 ** attempt)  # exponential backoff
            print(f"Switching model...\n")    
        if isinstance(response, str):
            return response
        first_item = response[0]
        if isinstance(first_item, dict) and "message" in first_item:
            return first_item["message"].get("content", str(first_item))
        if hasattr(first_item, "message"):
            return first_item.message.content
        return str(response)
            
                
        # Сценарий 2: Ответ пришел как строка
        if isinstance(response, str):
            return response
            
        # Сценарий 3: Стандартный объект OpenAI SDK
        if hasattr(response, "choices"):
            # Проверяем, не является ли choices списком (обычно это список)
            choices = response.choices
            if isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                # Проверяем, является ли message словарем или объектом
                if isinstance(first_choice, dict) and "message" in first_choice:
                    return first_choice["message"].get("content", "")
                if hasattr(first_choice, "message"):
                    # Проверяем внутренности message
                    msg = first_choice.message
                    if isinstance(msg, dict):
                        return msg.get("content", "")
                    if hasattr(msg, "content"):
                        return msg.content
            return response.choices[0].message.content
            
        # Сценарий 4: Ответ пришел как обычный словарь (dict)
        if isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                return choices[0]["message"]["content"]
            
        return str(response)
    
    else:
        raise ValueError(f"Неизвестный источник модели: {source}. Используйте 'ollama/' или 'openrouter/'")


def evaluate_model(model_tag: str) -> dict:
    """Прогоняем модель через набор задач и считаем accuracy"""
    passed = 0
    details = []

    print(f"Начинаем тестирование модели: {model_tag}...")

    if not EVAL_TASKS:
        return {"model": model_tag, "accuracy": 0, "passed": 0, "details": ["Список задач пуст"]}

    for task in EVAL_TASKS:
        try:
            # Вызываем нашу универсальную функцию
            answer = get_generation(model_tag, task["prompt"])
            
            # Проверяем ответ лямбдой из задачи
            success = task["check"](answer)

            if success:
                passed += 1
            details.append(f"  {'✓' if success else '✗'} {task['name']}")
        
        except Exception as e:
            # Если одна модель или один запрос упали — тест не прерывается
            details.append(f"  ✗ {task['name']} (Ошибка: {str(e)})")

    accuracy = (passed / len(EVAL_TASKS)) * 100
    return {
        "model": model_tag, 
        "accuracy": accuracy, 
        "passed": passed, 
        "details": details
    }


# Главный блок запуска бенчмарка
if __name__ == "__main__":
    print(f" Всего задач на тест: {len(EVAL_TASKS)}\n")
    
    for current_model in MODELS_TO_TEST:
        result = evaluate_model(current_model)
        
        if isinstance(result, dict):
            print(f"\nРезультат для {result.get('model', current_model)}:")
            print(f"Точность: {result.get('accuracy', 0):.0f}% ({result.get('passed', 0)}/{len(EVAL_TASKS)})")
            for detail in result.get('details', []):
                print(detail)
        else:
            # Если вернулась строка ошибки — просто печатаем её целиком
            print(f"\n❌ Ошибка при тестировании модели {current_model}:")
            print(result)

        print("-" * 40)


#Результат
#Результат для ollama/llama3.2:latest:
#Точность: 40% (8/20)
#  ✓ 1_Понимание формулировок1
#  ✓ 2_Понимание формулировок2
# ✗ 3_Понимание формулировок3
#  ✗ 4_Резюмирование проблемы
#  ✗ 5_Формулировка уточнения
# ✓ 6_Извлечение
#  ✓ 7_Классификация1
#  ✓ 8_Извлечение2
#  ✗ 9_Распознавание2
#  ✓ 10_Извлечение3
#  ✗ 11_Отсеивание мусора
#  ✗ 12_Извлечение4
#  ✗ 13_Классификация2
#  ✗ 14_Распознавание
#  ✓ 15_Оценка rритичности
#  ✗ 16_Распознавание взлома
#  ✗ 17_Нерелевантный запрос
#  ✗ 18_Противоречивый запрос
#  ✓ 19_Понимание разговорной речи
#  ✗ 20_Понимание разговорной речи2
#----------------------------------------
#Начинаем тестирование модели: ollama/qwen3:1.7b...
#
#Результат для ollama/qwen3:1.7b:
#Точность: 70% (14/20)
#  ✓ 1_Понимание формулировок1
#  ✓ 2_Понимание формулировок2
#  ✗ 3_Понимание формулировок3
#  ✗ 4_Резюмирование проблемы
#  ✓ 5_Формулировка уточнения
#  ✓ 6_Извлечение
#  ✓ 7_Классификация1
#  ✓ 8_Извлечение2
#  ✗ 9_Распознавание2
#  ✓ 10_Извлечение3
#  ✓ 11_Отсеивание мусора
#  ✓ 12_Извлечение4
#  ✓ 13_Классификация2
#  ✓ 14_Распознавание
#  ✓ 15_Оценка rритичности
#  ✗ 16_Распознавание взлома
#  ✗ 17_Нерелевантный запрос
#  ✓ 18_Противоречивый запрос
#  ✗ 19_Понимание разговорной речи
#  ✓ 20_Понимание разговорной речи2
#----------------------------------------
#Результат для openrouter/google/gemma-4-26b-a4b-it:free:
#Точность: 0% (0/20)
#  ✗ 1_Понимание формулировок1 (Ошибка: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "code": 400,\n    "message": "User location is not supported for the API use.",\n    "status": "FAILED_PRECONDITION"\n  }\n}\n', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 2_Понимание формулировок2 (Ошибка: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 3_Понимание формулировок3 (Ошибка: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 4_Резюмирование проблемы (Ошибка: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 5_Формулировка уточнения (Ошибка: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "code": 400,\n    "message": "User location is not supported for the API use.",\n    "status": "FAILED_PRECONDITION"\n  }\n}\n', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 6_Извлечение (Ошибка: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "code": 400,\n    "message": "User location is not supported for the API use.",\n    "status": "FAILED_PRECONDITION"\n  }\n}\n', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 7_Классификация1 (Ошибка: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "code": 400,\n    "message": "User location is not supported for the API use.",\n    "status": "FAILED_PRECONDITION"\n  }\n}\n', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 8_Извлечение2 (Ошибка: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "code": 400,\n    "message": "User location is not supported for the API use.",\n    "status": "FAILED_PRECONDITION"\n  }\n}\n', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 9_Распознавание2 (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 10_Извлечение3 (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 11_Отсеивание мусора (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 12_Извлечение4 (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 13_Классификация2 (Ошибка: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "code": 400,\n    "message": "User location is not supported for the API use.",\n    "status": "FAILED_PRECONDITION"\n  }\n}\n', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 14_Распознавание (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 15_Оценка rритичности (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 16_Распознавание взлома (Ошибка: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 17_Нерелевантный запрос (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 18_Противоречивый запрос (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 19_Понимание разговорной речи (Ошибка: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "code": 400,\n    "message": "User location is not supported for the API use.",\n    "status": "FAILED_PRECONDITION"\n  }\n}\n', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
#  ✗ 20_Понимание разговорной речи2 (Ошибка: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '16', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1779563580000'}, 'provider_name': None}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'})
