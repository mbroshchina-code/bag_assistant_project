users_per_day = 1648
questions_per_user = 3
input_tokens = 302
output_tokens = 250
days = 30

# доля кэшированных токенов (например, system prompt + история)
cache_ratio = 0.7  # 70% токенов переиспользуются

total_requests = users_per_day * questions_per_user * days  # 900,000
total_input = total_requests * input_tokens / 1_000_000     # 450M
total_output = total_requests * output_tokens / 1_000_000   # 270M

cached_input = total_input * cache_ratio
new_input = total_input * (1 - cache_ratio)

# цены ($/1M токенов)
providers = {
    "GPT-5.4": {
        "input": 2.50, "output": 15.00, "cached": 0.25
    },
    "Claude Sonnet 4.6": {
        "input": 3.00, "output": 15.00, "cached": 0.6
    },
    "Gemini 2.5 Pro": {
        "input": 1.25, "output": 10.00, "cached": 0.2
    },
    "GPT-5 mini": {
        "input": 0.25, "output": 1.00, "cached": 0.025
    },
    "Gemini Flash": {
        "input": 0.30, "output": 2.50, "cached": 0.03
    },
    "DeepSeek V3.2": {
        "input": 0.28, "output": 0.42, "cached": 0.28
    },
}

print(f"Запросов: {total_requests:,}")
print(f"Input: {total_input:.0f}M (cached: {cached_input:.0f}M)")
print(f"Output: {total_output:.0f}M\n")

for name, p in providers.items():
    cost = (
        new_input * p["input"] +
        cached_input * p["cached"] +
        total_output * p["output"]
    )
    print(f"{name:<22} ${cost:>10,.0f}")