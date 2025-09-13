"""
Скрипт для быстрой настройки универсального бота
"""
import os
import shutil
from utils import get_business_domain_examples, create_config_template

def setup_bot():
    """Интерактивная настройка бота"""
    print("🤖 **Настройка универсального Telegram-бота**\n")
    
    # Проверяем наличие .env файла
    if os.path.exists('.env'):
        response = input("📁 Файл .env уже существует. Перезаписать? (y/N): ")
        if response.lower() != 'y':
            print("✅ Настройка отменена")
            return
    
    print("🌐 **Выберите сферу деятельности:**")
    examples = get_business_domain_examples()
    
    domains = list(examples.keys())
    for i, domain in enumerate(domains, 1):
        print(f"   {i}. {domain}: {examples[domain]['description']}")
    
    print(f"   {len(domains) + 1}. Другая сфера (настройка вручную)")
    
    try:
        choice = int(input(f"\nВведите номер (1-{len(domains) + 1}): "))
        
        if choice == len(domains) + 1:
            # Ручная настройка
            domain = input("Введите название сферы: ").strip()
            keywords = input("Введите ключевые слова через запятую: ").strip()
            phrases = input("Введите фразы полного цикла через запятую: ").strip()
            
            config_template = f"""# Конфигурация для сферы: {domain}
BUSINESS_DOMAIN={domain}
BUSINESS_KEYWORDS={keywords}
FULL_CYCLE_PHRASES={phrases}

# Telegram API настройки
TELEGRAM_API_ID=your_api_id_here
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_PHONE=your_phone_number

# Целевые пользователи (ID через запятую)
TARGET_USER_IDS=user_id_1,user_id_2

# Настройки машинного обучения
ML_SIMILARITY_THRESHOLD=0.7
ML_MIN_TRAINING_EXAMPLES=3

# Фильтрация
FILTER_MIN_LENGTH=5
FILTER_BLACKLIST=спам,реклама
"""
        else:
            # Выбранная сфера
            domain = domains[choice - 1]
            config_template = create_config_template(domain)
            
    except (ValueError, IndexError):
        print("❌ Неверный выбор")
        return
    
    # Создаем .env файл
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(config_template)
        
        print(f"✅ Файл .env создан для сферы: {domain}")
        print("\n📝 **Следующие шаги:**")
        print("1. Отредактируйте файл .env")
        print("2. Заполните настройки Telegram API")
        print("3. Укажите целевых пользователей")
        print("4. Запустите бота: python main_universal.py")
        
    except Exception as e:
        print(f"❌ Ошибка создания файла .env: {e}")

def create_env_from_example():
    """Создает .env файл из примера"""
    if os.path.exists('env_example.txt'):
        try:
            shutil.copy('env_example.txt', '.env')
            print("✅ Файл .env создан из примера")
            print("📝 Отредактируйте файл .env и заполните настройки")
        except Exception as e:
            print(f"❌ Ошибка копирования файла: {e}")
    else:
        print("❌ Файл env_example.txt не найден")

def main():
    """Главная функция"""
    print("🚀 **Универсальный Telegram-бот - Настройка**\n")
    
    print("Выберите способ настройки:")
    print("1. Интерактивная настройка")
    print("2. Создать .env из примера")
    print("3. Выход")
    
    try:
        choice = int(input("\nВведите номер (1-3): "))
        
        if choice == 1:
            setup_bot()
        elif choice == 2:
            create_env_from_example()
        elif choice == 3:
            print("👋 До свидания!")
        else:
            print("❌ Неверный выбор")
            
    except ValueError:
        print("❌ Введите корректный номер")

if __name__ == "__main__":
    main()
