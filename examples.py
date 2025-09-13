"""
Примеры использования универсального бота для разных сфер
"""
from utils import get_business_domain_examples, create_config_template

def print_all_examples():
    """Выводит все примеры конфигурации"""
    examples = get_business_domain_examples()
    
    for domain, info in examples.items():
        print(f"\n🌐 **{domain.upper()}**")
        print(f"Описание: {info['description']}")
        print(f"Ключевые слова: {info['keywords']}")
        print(f"Фразы полного цикла: {info['full_cycle_phrases']}")
        print("-" * 50)

def create_config_files():
    """Создает файлы конфигурации для всех сфер"""
    examples = get_business_domain_examples()
    
    for domain in examples.keys():
        filename = f"config_{domain}.env"
        config_content = create_config_template(domain)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✅ Создан файл: {filename}")

def demo_text_analysis():
    """Демонстрация анализа текста"""
    from utils import clean_text, calculate_text_complexity, extract_keywords_from_text
    
    sample_texts = [
        "Нужен видеопродакшн полного цикла для рекламного ролика. От концепции до финального монтажа.",
        "Ищу фрилансера для создания логотипа. Бюджет ограничен.",
        "Требуется веб-разработка сайта под ключ. От дизайна до запуска.",
        "Предлагаю услуги фотографа. Свадебная фотосессия с обработкой.",
        "Нужен маркетолог для продвижения в соцсетях. Полный цикл от стратегии до реализации."
    ]
    
    print("🔍 **Демонстрация анализа текста:**\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"**Текст {i}:** {text}")
        
        # Очистка
        cleaned = clean_text(text)
        print(f"Очищенный: {cleaned}")
        
        # Сложность
        complexity = calculate_text_complexity(text)
        print(f"Сложность: {complexity['complexity']:.2f}")
        
        # Ключевые слова
        keywords = extract_keywords_from_text(text)
        print(f"Ключевые слова: {', '.join(keywords[:5])}")
        
        print("-" * 50)

def demo_ml_classifier():
    """Демонстрация работы классификатора"""
    print("🤖 **Демонстрация классификатора:**\n")
    
    try:
        from ml_classifier import UniversalMessageClassifier
        from database import DatabaseManager
        
        # Создаем классификатор
        db = DatabaseManager(':memory:')  # В памяти для демо
        classifier = UniversalMessageClassifier(db_manager=db)
        
        # Добавляем примеры обучения
        training_examples = [
            ("Нужен видеопродакшн полного цикла", 1),
            ("Ищу фрилансера для логотипа", 0),
            ("Требуется веб-разработка под ключ", 1),
            ("Предлагаю услуги фотографа", 0),
            ("Нужен маркетолог для продвижения", 1),
            ("Ищу дизайнера для баннера", 0)
        ]
        
        print("📚 Добавляем примеры обучения:")
        for text, label in training_examples:
            success = classifier.add_training_example(text, label)
            print(f"   {'✅' if success else '❌'} {text} -> {label}")
        
        # Обучаем модель
        print("\n🔄 Обучение модели...")
        success = classifier.auto_train()
        print(f"Обучение: {'✅' if success else '❌'}")
        
        # Тестируем
        test_texts = [
            "Нужен полный цикл видеопродакшна",
            "Ищу фрилансера",
            "Требуется комплексная разработка",
            "Предлагаю услуги"
        ]
        
        print("\n🎯 Тестирование:")
        for text in test_texts:
            probability = classifier.predict(text)
            prediction = "Релевантно" if probability and probability > 0.5 else "Нерелевантно"
            print(f"   {text} -> {prediction} ({probability:.3f})")
        
        # Статистика
        stats = classifier.get_stats()
        print(f"\n📊 Статистика:")
        print(f"   Обучена: {'✅' if stats['is_trained'] else '❌'}")
        print(f"   Примеров: {stats['training_examples']}")
        if stats.get('accuracy'):
            print(f"   Точность: {stats['accuracy']:.2%}")
        
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")

def main():
    """Главная функция с примерами"""
    print("📚 **ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ УНИВЕРСАЛЬНОГО БОТА**\n")
    
    print("Выберите пример:")
    print("1. Все примеры конфигурации")
    print("2. Создать файлы конфигурации")
    print("3. Демонстрация анализа текста")
    print("4. Демонстрация классификатора")
    print("5. Все примеры")
    
    try:
        choice = int(input("\nВведите номер (1-5): "))
        
        if choice == 1:
            print_all_examples()
        elif choice == 2:
            create_config_files()
        elif choice == 3:
            demo_text_analysis()
        elif choice == 4:
            demo_ml_classifier()
        elif choice == 5:
            print_all_examples()
            print("\n" + "="*60 + "\n")
            demo_text_analysis()
            print("\n" + "="*60 + "\n")
            demo_ml_classifier()
        else:
            print("❌ Неверный выбор")
            
    except ValueError:
        print("❌ Введите корректный номер")
    except KeyboardInterrupt:
        print("\n👋 До свидания!")

if __name__ == "__main__":
    main()
