import time
import ollama
import json


def benchmark_model(model: str, prompt: str, runs: int = 3) -> dict:
    """Полный бенчмарк с несколькими прогонами для усреднения"""
    results = []

    for i in range(runs):
        start = time.perf_counter()
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        ttft = None
        tokens = 0

        for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - start
            tokens += 1

        total = time.perf_counter() - start
        gen_time = total - ttft if ttft else total
        throughput = tokens / gen_time if gen_time > 0 else 0

        results.append({
            "ttft": ttft,
            "total": total,
            "tokens": tokens,
            "throughput": throughput
        })

    # Усредняем результаты
    avg = {
        "model": model,
        "avg_ttft": round(sum(r["ttft"] for r in results) / runs, 3),
        "avg_total": round(sum(r["total"] for r in results) / runs, 3),
        "avg_tokens": round(sum(r["tokens"] for r in results) / runs),
        "avg_throughput": round(sum(r["throughput"] for r in results) / runs, 1),
    }
    return avg


# Сравниваем три модели
models = ["qwen3.5:4b", "deepseek-r1:1.5b"]
prompt = "Напиши функцию на Python для сортировки списка словарей по ключу"

print("Бенчмарк моделей (среднее за 3 прогона):\n")
for model in models:
    result = benchmark_model(model, prompt)
    print(f"{result['model']:20s} | TTFT: {result['avg_ttft']:.3f}s | "
          f"Total: {result['avg_total']:.2f}s | {result['avg_throughput']:.1f} tok/s")


# Пример вывода:
# qwen3:1.7b           | TTFT: 0.185s | Total: 2.14s | 72.3 tok/s
# llama3.2:3b          | TTFT: 0.312s | Total: 3.87s | 45.1 tok/s
# gemma3:4b            | TTFT: 0.401s | Total: 4.52s | 38.7 tok/s