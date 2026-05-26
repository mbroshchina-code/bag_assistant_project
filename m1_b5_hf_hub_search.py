from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))

# Поиск моделей для генерации текста, отсортированных по скачиваниям
print("Топ-5 моделей для summarization:\n")
models = api.list_models(
    pipeline_tag="summarization",
    sort="downloads",
    limit=5
)

for model in models:
    print(f"  {model.id}")
    print(f"    Скачиваний: {model.downloads:,}")
    print(f"    Лайков: {model.likes:,}")
    print(f"    Теги: {', '.join(model.tags[:5]) if model.tags else 'нет'}")
    print()

# Получаем детальную информацию о конкретной модели
print("="*50)
print("Детали модели Qwen/Qwen3-ASR-1.7B:\n")
info = api.model_info("Qwen/Qwen3-ASR-1.7B")
print(f"  ID: {info.id}")
print(f"  Автор: {info.author}")
print(f"  Скачиваний: {info.downloads:,}")
print(f"  Лицензия: {info.card_data.get('license', 'не указана') if info.card_data else 'нет данных'}")
print(f"  Теги: {', '.join(info.tags[:8]) if info.tags else 'нет'}")
print(f"  Размер: {info.safetensors.total if info.safetensors else 'неизвестно'} параметров")
print("Детали модели meta-llama/Llama-3.2-1B-Instruct:\n")
info = api.model_info("meta-llama/Llama-3.2-1B-Instruct")
print(f"  ID: {info.id}")
print(f"  Автор: {info.author}")
print(f"  Скачиваний: {info.downloads:,}")
print(f"  Лицензия: {info.card_data.get('license', 'не указана') if info.card_data else 'нет данных'}")
print(f"  Теги: {', '.join(info.tags[:8]) if info.tags else 'нет'}")
print(f"  Размер: {info.safetensors.total if info.safetensors else 'неизвестно'} параметров")
print("Детали модели google/gemma-3-1b-it:\n")
info = api.model_info("google/gemma-3-1b-it")
print(f"  ID: {info.id}")
print(f"  Автор: {info.author}")
print(f"  Скачиваний: {info.downloads:,}")
print(f"  Лицензия: {info.card_data.get('license', 'не указана') if info.card_data else 'нет данных'}")
print(f"  Теги: {', '.join(info.tags[:8]) if info.tags else 'нет'}")
print(f"  Размер: {info.safetensors.total if info.safetensors else 'неизвестно'} параметров")


#Топ-5 моделей для summarization:
#
#  facebook/bart-large-cnn
#    Скачиваний: 1,947,975
#  v Лайков: 1,581
#    Теги: transformers, pytorch, tf, jax, rust
#
#  sshleifer/distilbart-cnn-12-6
#    Скачиваний: 1,341,593
#    Лайков: 319
#    Теги: transformers, pytorch, jax, rust, bart
#
#  IlyaGusev/rut5_base_headline_gen_telegram
#    Скачиваний: 609,358
#    Лайков: 9
#    Теги: transformers, pytorch, t5, text2text-generation, summarization
#
#  philschmid/bart-large-cnn-samsum
#    Скачиваний: 291,969
#    Лайков: 267
#    Теги: transformers, pytorch, bart, text2text-generation, sagemaker
#
#  google/pegasus-xsum
#    Скачиваний: 228,324
#    Лайков: 219
#    Теги: transformers, pytorch, tf, jax, pegasus

#Топ-5 моделей для text-generation:
#
#  Qwen/Qwen3-0.6B
#    Скачиваний: 19,010,442
#    Лайков: 1,268
#    Теги: transformers, safetensors, qwen3, text-generation, conversational
#
#  openai-community/gpt2
#    Скачиваний: 17,054,818
#    Лайков: 3,261
#    Теги: transformers, pytorch, tf, jax, tflite
#
#  Qwen/Qwen2.5-1.5B-Instruct
#    Скачиваний: 14,840,982
#    Лайков: 711
#    Теги: transformers, safetensors, qwen2, text-generation, chat
#
#  Qwen/Qwen2.5-7B-Instruct
#    Скачиваний: 13,499,212
#    Лайков: 1,299
#    Теги: transformers, safetensors, qwen2, text-generation, chat
#
#  Qwen/Qwen3-8B
#    Скачиваний: 12,815,410
#    Лайков: 1,103
#    Теги: transformers, safetensors, qwen3, text-generation, conversational

#==================================================
#Детали модели Qwen3-1.7B:
#
#  ID: Qwen/Qwen3-ASR-1.7B
#  Автор: Qwen
#  Скачиваний: 1,956,782
#  Лицензия: apache-2.0
#  Теги: safetensors, qwen3_asr, automatic-speech-recognition, arxiv:2601.21337, license:apache-2.0, eval-results, deploy:azure, region:us
#  Размер: 2349217408 параметров

#Детали модели Llama-3.2-1B-Instruct:
#ID: meta-llama/Llama-3.2-1B-Instruct
#  Автор: meta-llama
#  Скачиваний: 8,253,739
#  Лицензия: llama3.2
#  Теги: transformers, safetensors, llama, text-generation, facebook, meta, pytorch, llama-3
#  Размер: 1235814400 параметров

#Детали модели google/gemma-3-1b-it:
#
#  ID: google/gemma-3-1b-it
#  Автор: google
#  Скачиваний: 1,286,389
#  Лицензия: gemma
#  Теги: transformers, safetensors, gemma3_text, text-generation, conversational, arxiv:1905.07830, arxiv:1905.10044, arxiv:1911.11641
#  Размер: 999885952 параметров



