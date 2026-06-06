PROVIDERS = {
    #"1": {
    #    "name": "OpenRouter",
    #    "api_key": None,
    #    "base_url": "https://openrouter.ai/api/v1",
    #    "model": "gpt-oss-20b:free",
    #    "env_key": "OPENROUTER_API_KEY",
    #},
    "1": {
        "name": "OpenAI",
        "api_key": None,
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "2": {
        "name": "Google",
        "api_key": None,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "gemini-3.5-flash",
        "env_key": "OPENROUTER_API_KEY_GOOGLE",
    },
    "3": {
        "name": "Ollama (локальный)",
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:1.7b",
        "env_key": None,
    },
}
