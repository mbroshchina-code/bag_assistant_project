def calculate_cost(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_price_per_m: float,   # Цена за 1M input-токенов
    output_price_per_m: float,  # Цена за 1M output-токенов
    cache_ratio: float = 0.0,   # Доля кэшированных input-токенов (0.0–1.0)
    cache_discount: float = 0.9 # Скидка на кэшированные токены (0.9 = 90%)
) -> dict:
    """Калькулятор стоимости ИИ-проекта"""
    days_in_month = 30

    # Общее количество токенов в месяц
    total_input = requests_per_day * avg_input_tokens * days_in_month
    total_output = requests_per_day * avg_output_tokens * days_in_month

    # Стоимость input с учётом кэширования
    cached_input = total_input * cache_ratio
    uncached_input = total_input * (1 - cache_ratio)
    input_cost = (
        uncached_input * input_price_per_m / 1_000_000 +
        cached_input * input_price_per_m * (1 - cache_discount) / 1_000_000
    )

    # Стоимость output (не кэшируется)
    output_cost = total_output * output_price_per_m / 1_000_000

    total = input_cost + output_cost

    return {
        "requests_month": requests_per_day * days_in_month,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_cost": round(input_cost, 2),
        "output_cost": round(output_cost, 2),
        "total_monthly": round(total, 2),
        "cost_per_request": round(total / (requests_per_day * days_in_month), 6)
    }


# Сценарий: чат-бот с 4950 запросов/день
chatbot = calculate_cost(
    requests_per_day=4950,
    avg_input_tokens=492,  # system prompt 190+ запрос пользователя 302
    avg_output_tokens=380,  # ответ модели
    input_price_per_m=0.065, # qwen3.5-flash
    output_price_per_m=0.26,
    cache_ratio=0.7,        # 70% input — это system prompt (кэшируется)
    cache_discount=0.9      # 90% скидка на кэш у qwen3.5-flash
)


print("Чат-бот на qwen3.5-flash:")
print(f"  Запросов/месяц: {chatbot['requests_month']:,}")
print(f"  Input: ${chatbot['input_cost']}")
print(f"  Output: ${chatbot['output_cost']}")
print(f"  Итого: ${chatbot['total_monthly']}/месяц")
print(f"  За один запрос: ${chatbot['cost_per_request']}")
# Итого: ~$.../месяц за ... запросов


# Один сценарий: чат-бот, 4950 запросов/день, 492 input, 380 output
providers = [
    {"name": "GPT-5.4",           "input": 2.50,  "output": 15.00, "cache_discount": 0.90},
    {"name": "GPT-5 Mini",        "input": 0.25,  "output": 2.00,  "cache_discount": 0.90},
    {"name": "GPT-4.1 Nano",      "input": 0.10,  "output": 0.40,  "cache_discount": 0.75},
    {"name": "Claude Sonnet 4.6", "input": 3.00,  "output": 15.00, "cache_discount": 0.90},
    {"name": "Claude Haiku 4.5",  "input": 1.00,  "output": 5.00,  "cache_discount": 0.90},
    {"name": "Gemini 3 Flash",    "input": 0.50,  "output": 3.00,  "cache_discount": 0.50},
    {"name": "Gemini Flash-Lite", "input": 0.10,  "output": 0.40,  "cache_discount": 0.50},
    {"name": "DeepSeek V3.2",     "input": 0.28,  "output": 0.42,  "cache_discount": 0.90},
    {"name": "Gpt-oss-20b",       "input": 0.042,  "output": 0.160, "cache_discount": 0.00},
    {"name": "Llama 3.2 3B",      "input": 0.048,  "output": 0.324, "cache_discount": 0.00}, 
    {"name": "Qwen3.5-flash",     "input": 0.065,  "output": 0.26, "cache_discount": 0.90}, 
]

print(f"{'Провайдер':25s} | {'Без кэша':>10s} | {'С кэшем 70%':>12s} | {'За запрос':>10s}")
print("-" * 65)

for p in providers:
    # Без кэша
    no_cache = calculate_cost(4950, 492, 380, p["input"], p["output"])
    # С кэшем
    with_cache = calculate_cost(4950, 492, 380, p["input"], p["output"],
                                cache_ratio=0.7, cache_discount=p["cache_discount"])
    print(f"{p['name']:25s} | ${no_cache['total_monthly']:>8.2f} | "
          f"${with_cache['total_monthly']:>10.2f} | ${with_cache['cost_per_request']:.6f}")

# Пример вывода референса:
# GPT-5.4                   |   $240.00 |     $202.20 | $0.006740
# GPT-5 Mini                |    $30.00 |      $26.22 | $0.000874
# GPT-4.1 Nano              |     $7.20 |       $5.94 | $0.000198
# Claude Sonnet 4.6         |   $252.00 |     $206.64 | $0.006888
# Claude Haiku 4.5          |    $84.00 |      $68.88 | $0.002296
# Gemini 3 Flash            |    $48.00 |      $43.80 | $0.001460
# Gemini Flash-Lite         |     $7.20 |       $6.36 | $0.000212
# DeepSeek V3.2             |    $11.76 |       $7.53 | $0.000251

# Пример вывода МОЙ: 
# Чат-бот на qwen3.5-flash:
#  Запросов/месяц: 148,500
#  Input: $1.76
# Output: $14.67
#  Итого: $16.43/месяц
# За один запрос: $0.000111

# Провайдер                 |   Без кэша |  С кэшем 70% |  За запрос
#-----------------------------------------------------------------
# Провайдер                 |   Без кэша |  С кэшем 70% |  За запрос
#-----------------------------------------------------------------
# GPT-5.4                   | $ 1029.11 | $    914.03 | $0.006155
# GPT-5 Mini                | $  131.13 | $    119.62 | $0.000806
# GPT-4.1 Nano              | $   29.88 | $     26.04 | $0.000175
# Claude Sonnet 4.6         | $ 1065.64 | $    927.55 | $0.006246
# Claude Haiku 4.5          | $  355.21 | $    309.18 | $0.002082
# Gemini 3 Flash            | $  205.82 | $    193.04 | $0.001300
# Gemini Flash-Lite         | $   29.88 | $     27.32 | $0.000184
# DeepSeek V3.2             | $   44.16 | $     31.27 | $0.000211
# Gpt-oss-20b               | $   12.10 | $     12.10 | $0.000081
# Llama 3.2 3B              | $   21.79 | $     21.79 | $0.000147
# Qwen3.5-flash             | $   19.42 | $     16.43 | $0.000111