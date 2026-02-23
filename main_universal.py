import asyncio
import logging
import signal
import sys
from datetime import datetime

from config import config
from database import DatabaseManager
from ml_classifier import UniversalMessageClassifier
from telegram_bot import TelegramBot
from utils import validate_config, get_business_domain_examples

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s',
    handlers=[
        logging.FileHandler('universal_bot.log'),
        logging.StreamHandler()
    ]
)

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 УНИВЕРСАЛЬНЫЙ БОТ                      ║
║              Автоматическая фильтрация сообщений             ║
║                     с машинным обучением                    ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_config_info():
    print(f"📋 **Текущая конфигурация:**")
    print(f"   • Сфера: {config.business.business_domain}")
    print(f"   • Ключевых слов: {len(config.business.keywords)}")
    print(f"   • Целевых пользователей: {len(config.business.target_user_ids)}")
    print(f"   • Порог сходства: {config.ml.similarity_threshold}")
    print(f"   • Минимальная длина сообщения: {config.filter.min_message_length}")
    print()

def print_available_domains():
    print("🌐 **Доступные сферы деятельности:**")
    examples = get_business_domain_examples()
    for domain, info in examples.items():
        print(f"   • {domain}: {info['description']}")
    print()

def setup_configuration():
    print("🔧 **Настройка конфигурации:**")
    
    import os
    if not os.path.exists('.env'):
        print("❌ Файл .env не найден!")
        print("📝 Создайте файл .env на основе .env.example")
        print("   Скопируйте .env.example в .env и заполните настройки")
        return False
    errors = validate_config()
    if errors:
        print("❌ **Ошибки конфигурации:**")
        for error in errors:
            print(f"   • {error}")
        print()
        print("📝 Проверьте файл .env и исправьте ошибки")
        return False
    
    print("✅ Конфигурация корректна")
    return True

async def main():
    print_banner()
    
    if not setup_configuration():
        print("❌ Не удалось настроить конфигурацию. Завершение работы.")
        return
    
    print_config_info()
    
    bot = None

    def shutdown_handler():
        logging.info("🛑 Получен сигнал остановки, завершаю работу...")
        if bot and bot.client:
            bot.client.disconnect()

    loop = asyncio.get_event_loop()
    if sys.platform != 'win32':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)

    try:
        logging.info("🚀 Инициализация компонентов...")

        db_manager = DatabaseManager()
        logging.info("✅ База данных инициализирована")

        classifier = UniversalMessageClassifier(db_manager=db_manager)
        logging.info("✅ Классификатор инициализирован")

        bot = TelegramBot(db_manager=db_manager, classifier=classifier)
        logging.info("✅ Telegram бот инициализирован")

        stats = classifier.get_stats()
        logging.info(f"📊 Модель: {'обучена' if stats['is_trained'] else 'не обучена'}")
        logging.info(f"📚 Примеров для обучения: {stats['training_examples']}")

        if stats.get('accuracy'):
            logging.info(f"🎯 Точность модели: {stats['accuracy']:.2%}")

        logging.info("🚀 Запуск бота...")
        await bot.run()

    except KeyboardInterrupt:
        logging.info("🛑 Получен сигнал остановки")
    except Exception as e:
        logging.error(f"❌ Критическая ошибка: {e}")
        sys.exit(1)
    finally:
        if bot:
            await bot.stop()
        logging.info("👋 Завершение работы")

def print_help():
    help_text = """
🤖 **УНИВЕРСАЛЬНЫЙ TELEGRAM-БОТ**

**Описание:**
Автоматически фильтрует и пересылает релевантные сообщения из Telegram чатов
с использованием машинного обучения и семантического анализа.

**Возможности:**
• 🎯 Семантический анализ сообщений
• 🤖 Машинное обучение с автообучением
• 📊 Детальная статистика и метрики
• 🔧 Настраиваемые ключевые слова для любой сферы
• 💾 База данных SQLite для хранения данных
• 📈 Система обратной связи для улучшения точности

**Настройка:**
1. Скопируйте .env.example в .env
2. Заполните настройки Telegram API
3. Настройте ключевые слова под вашу сферу
4. Укажите целевых пользователей

**Команды бота:**
• /help - справка
• /stats - статистика модели и бота
• /train - переобучение модели
• /correct_<id> - отметить сообщение как релевантное
• /wrong_<id> - отметить сообщение как нерелевантное
• /clear_history - очистить старую историю

**Примеры сфер:**
• video_production - видеопродакшн
• web_development - веб-разработка
• design - дизайн и брендинг
• marketing - маркетинг и продвижение
• photography - фотография

**Запуск:**
python main_universal.py
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_help()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            sys.exit(1)
