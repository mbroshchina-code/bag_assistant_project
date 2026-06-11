import os
import time
import logging
import json
import hashlib
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError
from tenacity import retry, wait_exponential, wait_random, stop_after_attempt, retry_if_exception_type

# импортируем словарь PROVIDERS
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

class LLMCache:
    """In-memory кэш для ответов LLM с поддержкой TTL на SHA-256 и статистикой."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._cache: dict[str, tuple[str, float]] = {}
        self.ttl = ttl_seconds 
        self.hits = 0
        self.misses = 0
    
    def _load_cache(self) -> None:
        """Загружает кеш из локального файла, если он существует."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"[Cache] Загружено {len(self.cache)} записей из кеша.")
            except Exception as e:
                logger.warning(f"[Cache] Не удалось прочитать файл кеша: {e}")

    def _make_key(
        self, model: str, messages: list[dict], temperature: float = 0
    ) -> str:
        """Ключ = хеш(модель + параметры + промпт)."""
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], temperature: float = 0) -> str | None:
        # ИСПРАВЛЕНО: добавляем генерацию ключа, чтобы избежать NameError
        key = self._make_key(model, messages, temperature)
        print(f"   [DEBUG CACHE GET] Ключ: {key}") # ВРЕМЕННЫЙ ЛОГ
        
        if key in self._cache:
            value, created_at = self._cache[key]
            if time.time() - created_at < self.ttl:
                self.hits += 1
                logger.info("   [LLMCache] !!! HIT !!! Ответ успешно взят из памяти.")
                return value
            # TTL истёк — удаляем запись, чтобы не засорять RAM
            logger.info("   [LLMCache] Запись найдена, но просрочена по TTL. Удаляю.")
            del self._cache[key]  # TTL истёк
            
        self.misses += 1
        logger.info("   [LLMCache] !!! MISS !!! Запись отсутствует в памяти.")
        return None

    def set(self, model: str, messages: list[dict], temperature: float, response: str) -> None:
        key = self._make_key(model, messages, temperature)
        self._cache[key] = (response, time.time())
        print(f"   [DEBUG CACHE SET] Ключ: {key}") # ВРЕМЕННЫЙ ЛОГ
        
    def clear(self) -> None:
        """Полная очистка кэша и сброс метрик."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("   [LLMCache] Память кэша полностью очищена.")

    @property
    def hit_rate(self) -> float:
        """Вычисляет эффективность кэша в процентах."""
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0


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
        
        # Инициализируем наш новый обновленный кэш (TTL: 1 час)
        self.cache = LLMCache(ttl_seconds=3600)
        
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
    def _execute_request(self, client: OpenAI, model: str, temperature: float, messages: List[Dict[str, str]]) -> Any:
        """Вызов API с гарантией 5 ретраев при лимитах и таймаутах."""
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
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

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Последовательно перебирает провайдеров с проверкой Circuit Breaker и Rate Limiter."""
        now = time.time()
        # 1. Проверяем кэш по новой сигнатуре аргументов ДО работы с сетью
        if self.active_providers:
            target_model = self.active_providers[0]["model"]
            cached_response = self.cache.get(target_model, messages, temperature)
            if cached_response is not None:
                return cached_response

        # 2. Если Cache Miss — перебираем провайдеров (Fallback)
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
                    temperature,
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
            
                self.cache.set(provider["model"], messages, temperature, response_text)
                
                self.cost_tracker.log_usage(
                    model=provider["model"],
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )
                return response_text
            
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
    
# --------------------------------------------------------------------------
# ОБНОВЛЕННЫЙ МЕТОД КЛАССА RobustLLMClient (Внутри класса)
# --------------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Последовательно перебирает провайдеров (Fallback) с оптимизированным кэшем."""
        now = time.time()
        start_time = time.time()
        
        # Выделяем ОДНУ базовую модель для кэша (модель нашего главного приоритетного провайдера)
        base_cache_model = self.active_providers[0]["model"]

        
        # РЕАЛИЗАЦИЯ ВАШЕЙ ИДЕИ: Для кэша берём только system + последнее сообщение пользователя
        # Это предотвращает промахи кэша при росте истории диалога
        cache_messages = [m for m in messages if m["role"] == "system"]
        if messages and messages[-1]["role"] == "user":
            cache_messages.append(messages[-1])
        elif messages:
            cache_messages.append(messages[-1]) # На случай, если последнее сообщение не от user

        # 1. Проверяем кэш по оптимизированной структуре сообщений
        if self.active_providers:
            target_model = self.active_providers[0]["model"]
            cached_response = self.cache.get(base_cache_model, cache_messages, temperature)
            if cached_response is not None:
                elapsed = time.time() - start_time
                logger.info("Ответ успешно взят из кэша за %.4fs", elapsed)
                return cached_response

        # 2. Если Cache Miss — идем по списку доступных бэкендов (передаем ПОЛНУЮ историю для контекста)
        for provider in self.active_providers:
            if now < provider["circuit_open_until"]:
                continue

            try:
                self._apply_rate_limiter(provider)
                
                logger.info("Запрос к провайдеру: %s (модель: %s)", provider["name"], provider["model"])
                response = self._execute_request(
                    provider["client"],
                    provider["model"],
                    temperature,
                    messages, # Здесь идет ПОЛНАЯ история, чтобы модель помнила контекст диалога
                )
                
                provider["failures_count"] = 0
                provider["circuit_open_until"] = 0.0
                
                response_text = response.choices[0].message.content
                
                # 3. Сохраняем в кэш связку именно с оптимизированными cache_messages и ТЕКУЩЕЙ температурой
                self.cache.set(base_cache_model, cache_messages, temperature, response_text)
                
                self.cost_tracker.log_usage(
                    model=provider["model"],
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )
                
                elapsed = time.time() - start_time
                logger.info("API ответ получен успешно за %.2fs через %s", elapsed, provider["name"])
                return response_text
                
            except (APIError, APIConnectionError, APITimeoutError) as e:
                provider["failures_count"] += 1
                if provider["failures_count"] >= self.cb_max_failures:
                    provider["circuit_open_until"] = time.time() + self.cb_cooldown_period
                    logger.error("   [Circuit Breaker] Провайдер %s заблокирован на 60 сек.", provider["name"])
                continue

        raise RuntimeError("Ни один из настроенных провайдеров не ответил на запрос.")


# --------------------------------------------------------------------------
# ТОЧКА ВХОДА И ДЕМОНСТРАЦИЯ
# --------------------------------------------------------------------------
COMMANDS_HELP = "/cache — статистика | /clear_cache — очистка | exit — выход"

def main() -> None:
    llm = RobustLLMClient()
    print("Отправляю запрос через RobustLLMClient...\n")
    print(f"Доступные команды: {COMMANDS_HELP}\n")
    
    try:
        test_messages = [{"role": "user", "content": "Объясни в одном предложении, зачем нужен fallback."}]
        
        # Симулируем 3 запроса подряд (первый пойдет в сеть, остальные два — мгновенно в кэш)
        for i in range(1, 4):
            print(f"\n--- Вызов #{i} ---")
            answer = llm.chat(test_messages, temperature=0)
            print(f"Ответ: {answer}")
            
    except Exception as exc:
        print(f"Критическая ошибка выполнения: {exc}")
        
    # Демонстрируем работу вычисляемого свойства hit_rate после вызовов    
    print("\n--- Итоговые метрики кэширования ---")
    print(f"Успешных попаданий (Hits): {llm.cache.hits}")
    print(f"Промахов кэша (Misses): {llm.cache.misses}")
    print(f"Эффективность (Hit Rate): {llm.cache.hit_rate:.2f}%")


if __name__ == "__main__":
    main()


#Результаты:

#Отправляю запрос через RobustLLMClient...
#Доступные команды: /cache — статистика | /clear_cache — очистка | exit — выход
#--- Вызов #1 ---
#   [DEBUG CACHE GET] Ключ: f2bdae56b3d70dc4f77bb6d1404fd5efee222eb82e6448e15fe65293cd1f952d
#2026-06-12 00:07:50,240 - __main__ - INFO -    [LLMCache] !!! MISS !!! Запись отсутствует в памяти.
#2026-06-12 00:07:50,240 - __main__ - INFO - Запрос к провайдеру: OpenAI (модель: gpt-4o-mini)
#2026-06-12 00:07:51,501 - httpx - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 403 Forbidden"
#2026-06-12 00:07:51,505 - __main__ - INFO - Запрос к провайдеру: Google (модель: gemini-3.5-flash)
#2026-06-12 00:07:52,739 - httpx - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 402 Payment Required"
#2026-06-12 00:07:52,741 - __main__ - INFO - Запрос к провайдеру: Ollama (локальный) (модель: qwen3:1.7b)
#2026-06-12 00:08:22,758 - openai._base_client - INFO - Retrying request to /chat/completions in 0.375113 seconds
#2026-06-12 00:08:51,017 - httpx - INFO - HTTP Request: POST http://localhost:11434/v1/chat/completions "HTTP/1.1 200 OK"
#   [DEBUG CACHE SET] Ключ: f2bdae56b3d70dc4f77bb6d1404fd5efee222eb82e6448e15fe65293cd1f952d
#  [Токены: 26+312 | Стоимость: $0.000083 | За сессию: $0.000083]
#2026-06-12 00:08:51,027 - __main__ - INFO - API ответ получен успешно за 60.79s через Ollama (локальный)
#Ответ: Фallback нужен, чтобы обеспечить продолжительность работы системы и предотвратить сбои при возникновении ошибок или непредвиденных условий.
#
#--- Вызов #2 ---
#   [DEBUG CACHE GET] Ключ: f2bdae56b3d70dc4f77bb6d1404fd5efee222eb82e6448e15fe65293cd1f952d
#2026-06-12 00:08:51,028 - __main__ - INFO -    [LLMCache] !!! HIT !!! Ответ успешно взят из памяти.
#2026-06-12 00:08:51,028 - __main__ - INFO - Ответ успешно взят из кэша за 0.0010s
#Ответ: Фallback нужен, чтобы обеспечить продолжительность работы системы и предотвратить сбои при возникновении ошибок или непредвиденных условий.
#
#--- Вызов #3 ---
#   [DEBUG CACHE GET] Ключ: f2bdae56b3d70dc4f77bb6d1404fd5efee222eb82e6448e15fe65293cd1f952d
#2026-06-12 00:08:51,028 - __main__ - INFO -    [LLMCache] !!! HIT !!! Ответ успешно взят из памяти.
#2026-06-12 00:08:51,028 - __main__ - INFO - Ответ успешно взят из кэша за 0.0000s
#Ответ: Фallback нужен, чтобы обеспечить продолжительность работы системы и предотвратить сбои при возникновении ошибок или непредвиденных условий.
#
#--- Итоговые метрики кэширования ---
#Успешных попаданий (Hits): 2
#Промахов кэша (Misses): 1
#Эффективность (Hit Rate): 66.67%