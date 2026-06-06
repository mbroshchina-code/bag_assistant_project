import os
import time
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError
from tenacity import retry, wait_exponential, wait_random, stop_after_attempt, retry_if_exception_type

# Гарантированно импортируем ваш словарь PROVIDERS
from m2_b1_config import PROVIDERS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Конфигурация цен (за 1 млн токенов)
# --------------------------------------------------------------------------
PRICES_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "gpt-oss-20b:free": {"input": 0.042, "output": 0.160},       # Из конфига
    "gemini-3.5-flash": {"input": 0.505, "output": 8.97},        # Из конфига Gemini 
    "qwen3:1.7b": {"input": 0.065,  "output": 0.26},             # Локальная модель из конфига
}


# --------------------------------------------------------------------------
# Трекер стоимости за сессию
# --------------------------------------------------------------------------
class SessionCostTracker:
    """Подсчитывает общую стоимость всех запросов за сессию."""

    def __init__(self) -> None:
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0

    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Записывает usage одного запроса и выводит стоимость."""
        price = PRICES_PER_1M_TOKENS.get(model, {"input": 1.00, "output": 3.00})

        cost_input = prompt_tokens / 1_000_000 * price["input"]
        cost_output = completion_tokens / 1_000_000 * price["output"]
        cost = cost_input + cost_output

        self.total_cost += cost
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.request_count += 1

        print(
            f"  [Токены: {prompt_tokens}+{completion_tokens} | "
            f"Стоимость: ${cost:.6f} | "
            f"За сессию: ${self.total_cost:.6f}]"
        )


# --------------------------------------------------------------------------
# Основной клиент работы с LLM
# --------------------------------------------------------------------------
class RobustLLMClient:
    def __init__(self) -> None:
        self.raw_config: Dict[str, Dict[str, Any]] = PROVIDERS
        self.active_providers: List[Dict[str, Any]] = []
        self.cost_tracker = SessionCostTracker()
        
        # Настройки паттернов стабильности
        self.cb_max_failures = 3          # 3 ошибки подряд для блокировки
        self.cb_cooldown_period = 60.0    # Блокировка на 60 секунд
        self.min_time_between_requests = 1.0  # Клиентский лимит: минимум 1 сек между запросами (1 RPS)
        
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Динамически создает экземпляры OpenAI клиентов и инициализирует состояние CB/Rate Limiter."""
        for key, info in self.raw_config.items():
            api_key = info["api_key"]
            if api_key is None and info["env_key"]:
                api_key = os.getenv(info["env_key"])
            
            if not api_key and info["base_url"] and "openrouter.ai" in info["base_url"]:
                logger.warning(f"Пропущен API-ключ для {info['name']}. Провайдер будет пропущен.")
                continue

            client = OpenAI(
                api_key=api_key or "no-key-needed", 
                base_url=info["base_url"]
            )
            
            self.active_providers.append({
                "name": info["name"],
                "model": info["model"],
                "client": client,
                # Состояние Circuit Breaker
                "failures_count": 0,
                "circuit_open_until": 0.0,
                # Состояние Rate Limiter
                "last_request_time": 0.0
            })

        if not self.active_providers:
            raise RuntimeError("Критическая ошибка: Ни один провайдер не был успешно инициализирован.")

    @retry(
        wait=wait_exponential(multiplier=1, exp_base=2, min=1, max=16) + wait_random(0, 1),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        reraise=True  
    )
    def _execute_request(self, client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Any:
        """Вызов API с гарантией 5 ретраев при лимитах и таймаутах."""
        return client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=30,
        )

    def _apply_rate_limiter(self, provider: Dict[str, Any]) -> None:
        """Контролирует частоту отправки запросов на стороне клиента (Rate Limiter)."""
        now = time.time()
        elapsed = now - provider["last_request_time"]
        
        if elapsed < self.min_time_between_requests:
            sleep_needed = self.min_time_between_requests - elapsed
            logger.info(f"   [Rate Limiter] Замедление запроса к {provider['name']} на {sleep_needed:.2f} сек...")
            time.sleep(sleep_needed)
            
        provider["last_request_time"] = time.time()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Последовательно перебирает провайдеров с проверкой Circuit Breaker и Rate Limiter."""
        now = time.time()
        
        for provider in self.active_providers:
            # 1. Проверка Circuit Breaker
            if now < provider["circuit_open_until"]:
                remaining_block = provider["circuit_open_until"] - now
                logger.warning(
                    f"   [Circuit Breaker] Провайдер {provider['name']} временно заблокирован. "
                    f"Пропуск. До разблокировки: {remaining_block:.1f} сек."
                )
                continue

            try:
                # 2. Применение Rate Limiter
                self._apply_rate_limiter(provider)
                
                logger.info(f"Запрос к провайдеру: {provider['name']} (модель: {provider['model']})")
                response = self._execute_request(
                    provider["client"],
                    provider["model"],
                    messages,
                )
                
                # Если запрос успешен — полностью сбрасываем счетчик ошибок Circuit Breaker
                provider["failures_count"] = 0
                provider["circuit_open_until"] = 0.0
                
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
                logger.info(f"Успешно через {provider['name']}.")
                
                self.cost_tracker.log_usage(
                    model=provider["model"],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                return response.choices[0].message.content
                
            except (APIError, APIConnectionError, APITimeoutError) as e:
                # 3. Фиксация ошибки в Circuit Breaker (сюда попадаем, если упали все 5 попыток retry)
                provider["failures_count"] += 1
                logger.warning(
                    f"Провайдер {provider['name']} вернул ошибку ({provider['failures_count']}/{self.cb_max_failures}): {e}"
                )
                
                if provider["failures_count"] >= self.cb_max_failures:
                    provider["circuit_open_until"] = time.time() + self.cb_cooldown_period
                    logger.error(
                        f"   [Circuit Breaker] !!! ВЫКЛЮЧАТЕЛЬ РАЗОМКНУТ !!! "
                        f"Провайдер {provider['name']} заблокирован на {self.cb_cooldown_period} сек."
                    )
                continue

        raise RuntimeError("Ни один из настроенных провайдеров не доступен или все находятся в состоянии блокировки.")


def main() -> None:
    llm = RobustLLMClient()
    print("Отправляю запрос через RobustLLMClient...\n")
    try:
        # Демонстрационный цикл для проверки работы механизмов
        for i in range(1, 3):
            print(f"\n--- Шаг цикла #{i} ---")
            answer = llm.chat([
                {"role": "user", "content": "Объясни в одном предложении, зачем нужен fallback."}
            ])
            print(f"Ответ: {answer}")
    except Exception as exc:
        print(f"Критическая ошибка выполнения: {exc}")


if __name__ == "__main__":
    main()



#Резульат кода с логами (без Circuit breaker и  Rate limiter):
#2026-06-06 15:51:02,873 - __main__ - WARNING - Пропущен API-ключ для OpenAI. Провайдер будет пропущен.
#Отправляю запрос через RobustLLMClient...
#
#2026-06-06 15:51:03,070 - __main__ - INFO - Запрос к провайдеру: Google (модель: gemini-3.5-flash)
#2026-06-06 15:51:05,828 - httpx - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 402 Payment Required"
#2026-06-06 15:51:05,844 - __main__ - WARNING - Провайдер Google исчерпал 5 попыток и недоступен: Error code: 402 - {'error': {'message': "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 61723. To increase, visit https://openrouter.ai/workspaces/default/keys/693a9d61f2c6d9998705a1872de520d06db32b090f82296d6fa0ca4acb496277 and adjust the key's total limit", 'code': 402, 'metadata': {'provider_name': None, 'previous_errors': [{'code': 402, 'message': "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 61723. To increase, visit https://openrouter.ai/workspaces/default/keys/693a9d61f2c6d9998705a1872de520d06db32b090f82296d6fa0ca4acb496277 and adjust the key's total limit"}]}}, 'user_id': 'org_38VKohU3XUa4IpKe8EWDOYO737f'}
#2026-06-06 15:51:05,845 - __main__ - INFO - Запрос к провайдеру: Ollama (локальный) (модель: qwen3:1.7b)
#2026-06-06 15:51:35,872 - openai._base_client - INFO - Retrying request to /chat/completions in 0.491147 seconds
#2026-06-06 15:52:06,390 - openai._base_client - INFO - Retrying request to /chat/completions in 0.994693 seconds
#2026-06-06 15:52:59,883 - httpx - INFO - HTTP Request: POST http://localhost:11434/v1/chat/completions "HTTP/1.1 200 OK"
#2026-06-06 15:52:59,904 - __main__ - INFO - Успешно через Ollama (локальный).
#  [Токены: 26+190 | Стоимость: $0.000051 | За сессию: $0.000051]
#
#Ответ: Фallback нужен, чтобы обеспечить продолжение работы системы или процесса при возникновении ошибки или непредвиденного сбоя, предотвращая полное прекращение функционирования.  

#Резульат кода с логами с Circuit breaker и  Rate limiter:
#Отправляю запрос через RobustLLMClient...
#
#--- Шаг цикла #1 ---
#2026-06-06 16:14:01,077 - __main__ - INFO - Запрос к провайдеру: OpenAI (модель: gpt-4o-mini)
#2026-06-06 16:14:03,142 - httpx - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 403 Forbidden"
#2026-06-06 16:14:03,158 - __main__ - WARNING - Провайдер OpenAI вернул ошибку (1/3): Error code: 403 - {'error': {'code': 'unsupported_country_region_territory', 'message': 'Country, region, or territory not supported', 'param': None, 'type': 'request_forbidden'}}
#2026-06-06 16:14:03,158 - __main__ - INFO - Запрос к провайдеру: Google (модель: gemini-3.5-flash)
#2026-06-06 16:14:03,792 - httpx - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 402 Payment Required"
#2026-06-06 16:14:03,809 - __main__ - WARNING - Провайдер Google вернул ошибку (1/3): Error code: 402 - {'error': {'message': "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 61723. To increase, visit https://openrouter.ai/workspaces/default/keys/693a9d61f2c6d9998705a1872de520d06db32b090f82296d6fa0ca4acb496277 and adjust the key's total limit", 'code': 402, 'metadata': {'provider_name': None, 'previous_errors': [{'code': 402, 'message': "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 61723. To increase, visit https://openrouter.ai/workspaces/default/keys/693a9d61f2c6d9998705a1872de520d06db32b090f82296d6fa0ca4acb496277 and adjust the key's total limit"}]}}, 'user_id': 'org_38VKohU3XUa4IpKe8EWDOYO737f'}
#2026-06-06 16:14:03,809 - __main__ - INFO - Запрос к провайдеру: Ollama (локальный) (модель: qwen3:1.7b)
#2026-06-06 16:14:33,875 - openai._base_client - INFO - Retrying request to /chat/completions in 0.420120 seconds
#2026-06-06 16:14:55,738 - httpx - INFO - HTTP Request: POST http://localhost:11434/v1/chat/completions "HTTP/1.1 200 OK"
#2026-06-06 16:14:55,747 - __main__ - INFO - Успешно через Ollama (локальный).
#  [Токены: 26+243 | Стоимость: $0.000065 | За сессию: $0.000065]
#Ответ: Фallback нужен, чтобы обеспечить последующую работу системы или процесса в случае сбоев, обеспечивая устойчивость и предотвращая полную неисправимость.

#--- Шаг цикла #2 ---
#2026-06-06 16:14:55,748 - __main__ - INFO - Запрос к провайдеру: OpenAI (модель: gpt-4o-mini)
#2026-06-06 16:14:55,841 - httpx - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 403 Forbidden"
#2026-06-06 16:14:55,842 - __main__ - WARNING - Провайдер OpenAI вернул ошибку (2/3): Error code: 403 - {'error': {'code': 'unsupported_country_region_territory', 'message': 'Country, region, or territory not supported', 'param': None, 'type': 'request_forbidden'}}
#2026-06-06 16:14:55,842 - __main__ - INFO - Запрос к провайдеру: Google (модель: gemini-3.5-flash)
#2026-06-06 16:14:56,209 - httpx - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 402 Payment Required"
#2026-06-06 16:14:56,209 - __main__ - WARNING - Провайдер Google вернул ошибку (2/3): Error code: 402 - {'error': {'message': "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 61723. To increase, visit https://openrouter.ai/workspaces/default/keys/693a9d61f2c6d9998705a1872de520d06db32b090f82296d6fa0ca4acb496277 and adjust the key's total limit", 'code': 402, 'metadata': {'provider_name': None, 'previous_errors': [{'code': 402, 'message': "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 61723. To increase, visit https://openrouter.ai/workspaces/default/keys/693a9d61f2c6d9998705a1872de520d06db32b090f82296d6fa0ca4acb496277 and adjust the key's total limit"}]}}, 'user_id': 'org_38VKohU3XUa4IpKe8EWDOYO737f'}
#2026-06-06 16:14:56,210 - __main__ - INFO - Запрос к провайдеру: Ollama (локальный) (модель: qwen3:1.7b)
#2026-06-06 16:15:23,117 - httpx - INFO - HTTP Request: POST http://localhost:11434/v1/chat/completions "HTTP/1.1 200 OK"
#2026-06-06 16:15:23,118 - __main__ - INFO - Успешно через Ollama (локальный).
#  [Токены: 26+306 | Стоимость: $0.000081 | За сессию: $0.000146]
#Ответ: Фallback нужен, чтобы обеспечить стабильность и надежность системы, прибегая к альтернативному решению при отказе основного метода, предотвращая разрывы и сбояы.