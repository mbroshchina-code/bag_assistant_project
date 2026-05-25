from openai import OpenAI
import time
import os
from dotenv import load_dotenv
load_dotenv()

from real_eval_tasks import EVAL_TASKS
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY_GOOGLE")

def model_selection_pipeline(
    models: list,
    eval_tasks: list,
    speed_prompt: str,
    budget_per_month: float,  # Бюджет в долларах
    requests_per_day: int,
    avg_input: int,
    avg_output: int,
):
    """Полный pipeline: eval → benchmark → cost → рекомендация"""
    candidates = []
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    for m in models:
        print(f"\n{'=' * 40}\nТестирую: {m['name']}...")

        # 2. Speed benchmark
        start = time.perf_counter()
        stream = client.chat.completions.create(
            model=m["test_model"],
            messages=[{"role": "user", "content": speed_prompt}],
            stream=True
        )
        ttft = None
        tokens = 0
        for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - start
            tokens += 1
        total = time.perf_counter() - start

        # 3. Cost calculation (облачная цена модели)
        monthly_cost = (
            requests_per_day * 30 * avg_input * m["input_price"] / 1_000_000
            + requests_per_day * 30 * avg_output * m["output_price"] / 1_000_000
        )

        fits_budget = monthly_cost <= budget_per_month

        candidates.append(
            {
                "name": m["name"],
                "ttft": round(ttft, 3) if ttft else None,
                "throughput": (
                    round(tokens / (total - ttft), 1) if ttft and total > ttft else 0
                ),
                "monthly_cost": round(monthly_cost, 2),
                "fits_budget": fits_budget,
                "verdict": "✓" if fits_budget and m["accuracy"] >= 80 else "✗",
            }
        )

    # Рекомендация: самая дешёвая модель с accuracy >= 80% в рамках бюджета
    viable = [c for c in candidates if c["fits_budget"] and m['accuracy'] >= 80]
    viable.sort(key=lambda x: x["monthly_cost"])

    print(f"\n{'=' * 50}")
    print(
        f"{'Модель':20s} | {'Качество':>8s} | {'TTFT':>6s} | {'$/мес':>8s} | {'ОК':>3s}"
    )
    print(f"{'-' * 50}")
    for c in candidates:
        print(
            f"{c['name']:20s} | {m['accuracy']:>7.0f}% | {c['ttft']:>5.3f}s | "
            f"${c['monthly_cost']:>7.2f} | {c['verdict']}"
        )

    if viable:
        print(
            f"\n→ Рекомендация: {viable[0]['name']} "
            f"(${viable[0]['monthly_cost']}/мес, {viable[0]['accuracy']:.0f}% accuracy)"
        )


# Пример моделей (free-версии для тестирования, платные облачные цены для расчёта)
models_to_test = [
    {
        "name": "google/gemini-3.5-flash", # "gemma-4-26b-a4b-it:free",
        "test_model": "google/gemini-3.5-flash",
        "input_price": 0.505, # 0,118 "google/gemma-4-26b-a4b-it",
        "output_price": 8.97, # 0.418,
        "accuracy": 70,
    },
    #{
    #   "name": "google/gemma-4-31b-it:free",
    #   "test_model": "google/gemma-4-31b-it:free",
    #   "input_price": 0.146, #"google/gemma-4-31b-it"
    #   "output_price": 0.442,
    #   "accuracy": 70,
    #},
]

model_selection_pipeline(
    models=models_to_test,
    eval_tasks=EVAL_TASKS,
    speed_prompt="Напиши функцию на Python для валидации email",
    budget_per_month=100.0,
    requests_per_day=4950,
    avg_input=492,
    avg_output=380,
)

#Результат:
#Тестирую: google/gemini-3.5-flash...

#==================================================
#Модель                  | Качество |   TTFT |    $/мес |  ОК
#--------------------------------------------------
#google/gemini-3.5-flash |      70% | 5.211s | $ 543.07 | ✗