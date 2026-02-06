import os
import torch
import asyncio
from fastapi import FastAPI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from transformers import pipeline

# 1. Инициализация FastAPI
app = FastAPI()

# 2. Загрузка ИИ (TinyLlama)
print("--- LOADING AI ---")
# Используем максимально легкие настройки для CPU
pipe = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    device=-1  # Явно говорим использовать CPU
)

TOKEN = "8544814653:AAEvF5reDuvnXr7E0nuXaAL4uYUn-dj9KdI"

# 3. Логика ответов ИИ
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
        
    user_text = update.message.text
    print(f"User wrote: {user_text}")
    
    prompt = f"<|system|>\nТы — полезный ИИ помощник.</s>\n<|user|>\n{user_text}</s>\n<|assistant|>\n"
    
    # Генерация
    loop = asyncio.get_event_loop()
    # Запускаем тяжелую модель в отдельном потоке, чтобы бот не «замерзал»
    outputs = await loop.run_in_executor(None, lambda: pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7))
    
    answer = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
    await update.message.reply_text(answer)

# 4. Правильный запуск бота
@app.on_event("startup")
async def startup_event():
    print("--- STARTING BOT ---")
    application = Application.builder().token(TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Инициализация и старт
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    app.state.tg_app = application
    print("--- SYSTEM ONLINE ---")

@app.on_event("shutdown")
async def shutdown_event():
    application = app.state.tg_app
    if application:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

@app.get("/")
def home():
    return {"status": "AI Server is running"}
