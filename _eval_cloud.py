import os
from openai import OpenAI
import ollama
from dotenv import load_dotenv
from real_eval_tasks import EVAL_TASKS
import time

load_dotenv()

# Настраиваем клиент для облачных моделей OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = None

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# СПИСОК ВСЕХ МОДЕЛЕЙ ДЛЯ СРАВНЕНИЯ

MODELS_TO_TEST = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    #"deepseek/deepseek-v4-flash:free",
    #"openai/gpt-oss-20b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",            
]

def evaluate_model(model: str) -> dict:
    """Прогоняем модель через набор задач и считаем accuracy"""
    passed = 0
    details = []

    for task in EVAL_TASKS:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            temperature=0,
        )
        answer = response.choices[0].message.content
        success = task["check"](answer)

        if success:
            passed += 1
        details.append(f"  {'✓' if success else '✗'} {task['name']}")

    accuracy = passed / len(EVAL_TASKS) * 100
    return {"model": model, "accuracy": accuracy, "details": details}


# Оценка

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

#Результат по 20 задачам облачная модель (для локальных отдельный файл eval_local):
#Результат для openai/gpt-oss-20b:free:
#Точность: 60% (0/20)
#  ✓ 1_Понимание формулировок1
#  ✓ 2_Понимание формулировок2
#  ✗ 3_Понимание формулировок3
#  ✓ 4_Резюмирование проблемы
#  ✗ 5_Формулировка уточнения
#  ✓ 6_Извлечение
#  ✓ 7_Классификация1
#  ✓ 8_Извлечение2
#  ✗ 9_Распознавание2
#  ✓ 10_Извлечение3
#  ✗ 11_Отсеивание мусора
#  ✗ 12_Извлечение4
#  ✓ 13_Классификация2
#  ✓ 14_Распознавание
#  ✓ 15_Оценка rритичности
#  ✗ 16_Распознавание взлома
#  ✗ 17_Нерелевантный запрос
#  ✓ 18_Противоречивый запрос
#  ✓ 19_Понимание разговорной речи
#  ✗ 20_Понимание разговорной речи2


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

# Результат для openai/gpt-oss-20b:free:
# File "C:\Users\m.roschina\AppData\Local\Programs\Python\Python311\Lib\site-packages\openai\_base_client.py", line 1105, in request
#    raise self._make_status_error_from_response(err.response) from None
# openai.RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'openai/gpt-oss-20b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'OpenInference', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'}

# Результат для deepseek/deepseek-v4-flash:free:
# File "C:\Users\m.roschina\AppData\Local\Programs\Python\Python311\Lib\site-packages\openai\_base_client.py", line 1105, in request
#    raise self._make_status_error_from_response(err.response) from None
# openai.APIStatusError: Error code: 402 - {'error': {'message': 'Provider returned error', 'code': 402, 'metadata': {'raw': '{"error":{"type":"insufficient_quota","code":"insufficient_quota","message":"Out of credits. Top up at /dashboard/billing to continue.","request_id":"req_69SWcQNVNAoENei7"}}', 'provider_name': 'Crucible', 'is_byok': False}}, 'user_id': 'user_3DaAYF7ByXXxR6yTJfys0LcehB8'}
