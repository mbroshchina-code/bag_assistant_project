import ollama
import tiktoken

# модель
MODEL = "gemini_2.5_pro"

# цены за 1M токенов
PRICING = {
    "input": 1.25,
    "output": 10.00,
    "cached": 0.125  # обычно дешевле input
}

# encoder = tiktoken.get_encoding("cl100k_base")  # tokenizer
encoder = tiktoken.encoding_for_model("gpt-5")

def count_tokens(text):
    return len(encoder.encode(text))

# пример диалога
system_prompt = "Ты — виртуальный ассистент операторов технической поддержки по имени Эва. Эва с радостью поможет с поиском бага, подходящего под проблему клиента. Эва всегда отвечает на русском языке, вежливо и в деловом стиле. Эва не обсуждает сторонние вопросы, не относящиеся к багам и ошибкам. Эва использует в ответе всю полученную информацию по похожим на проблему клиента багам: номер, статус проблемы, дату создания, наименование проблемы, тематику, временное решение. Например, точно ты знаешь, что если пользователь пишет, что у него не проходит оплата картой/ Ошибка связи с сервером при оплате безналом, и подключен эквайринг - ты понимаешь, и отвечаешь, что это похоже на такой баг- Номер 2353909; Статус: закрыта; Создано: 28.04.2026 09:02; Наименование проблемы:Не проходит оплата с эквайрингом от Альфа банка. Ошибка нет связи с сервером; Тематика: Ошибки оплаты; Временное решение: Использовать другой способ оплаты. Ты не даешь никакой другой дополнительной информации и инструкций"
history = "Пользователь: Что мне делать, оплата картой не проходит?\nАссистент: Уточните, появляется ли ошибка при оплате?"
user_query = "Пользователь: пишет ошибка связи с сервером"

input_tokens = count_tokens(system_prompt + history + user_query)
output_tokens = 250  # допустим, модель ответила на 250 токенов

# допустим 70% истории закэшировано
cached_tokens = int(input_tokens * 0.7)
new_input_tokens = input_tokens - cached_tokens

# стоимость
cost = (
    new_input_tokens / 1_000_000 * PRICING["input"] +
    cached_tokens / 1_000_000 * PRICING["cached"] +
    output_tokens / 1_000_000 * PRICING["output"]
)

print(f"Input tokens: {input_tokens}")
print(f"Cached tokens: {cached_tokens}")
print(f"New input tokens: {new_input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Cost: ${cost:.6f}")

#Результат для MODEL = "gpt-5.4":
#Выполнение задачи: C:\Users\m.roschina\AppData\Local\Programs\Python\Python311\python.exe c:\Users\m.roschina\Desktop\Эвотор\мое\Обучение\bag_assistant_project\measure_token.py 

#Input tokens: 302
#Cached tokens: 211
#New input tokens: 91
#Output tokens: 250
#Cost: $0.004083

#Результат для MODEL = "gemini_2.5_pro": 
#Input tokens: 302
#Cached tokens: 211
#New input tokens: 91
#Output tokens: 250
#Cost: $0.002640

#Результат для MODEL = "gpt-5_mini":
#Input tokens: 302
#Cached tokens: 211
#New input tokens: 91
#Output tokens: 250
#Cost: $0.000278