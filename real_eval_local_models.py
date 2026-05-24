import ollama

from real_eval_tasks import EVAL_TASKS

# Чтобы код понимал, где какая модель, добавим к ним префиксы 'ollama/' и 'openrouter/'
MODELS_TO_TEST = [
    "ollama/llama3.2:latest", 
    "ollama/qwen3:1.7b",
]

def evaluate_model(model: str) -> dict:
    """Прогоняем модель через набор задач и считаем accuracy"""
    passed = 0
    details = []

    for task in EVAL_TASKS:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            temperature=0,
        )
        answer = response["message"]["content"]
        success = task["check"](answer)

        if success:
            passed += 1
        details.append(f"  {'✓' if success else '✗'} {task['name']}")

    accuracy = passed / len(EVAL_TASKS) * 100
    return {"model": model, "accuracy": accuracy, "details": details}


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


#Результат для локальных (для облачных отдельный файл _eval_cloud.py)
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
