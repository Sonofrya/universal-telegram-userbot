# 🤖 Universal Telegram Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Telethon](https://img.shields.io/badge/Telethon-1.28+-green.svg)](https://github.com/LonamiWebs/Telethon)

Автоматическая фильтрация и пересылка релевантных сообщений из Telegram чатов с использованием машинного обучения и семантического анализа.

> 🌟 **Универсальный бот для любой сферы деятельности** - от видеопродакшна до веб-разработки

## ✨ Возможности

- 🎯 **Семантический анализ** - поиск релевантных сообщений по смыслу
- 🤖 **Машинное обучение** - автообучение на основе обратной связи
- 📊 **Детальная статистика** - метрики точности и производительности
- 🔧 **Универсальность** - настройка под любую сферу деятельности
- 💾 **База данных** - SQLite для хранения данных и метрик
- 📈 **Система обратной связи** - улучшение точности через коррекции
- 🛡️ **Фильтрация** - черные списки и умные фильтры

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/yourusername/universal-telegram-bot.git
cd universal-telegram-bot
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Настройка конфигурации

#### Автоматическая настройка:
```bash
python setup.py
```

#### Ручная настройка:
```bash
cp env_example.txt .env
# Отредактируйте .env файл
```

Пример `.env`:
```env
TELEGRAM_API_ID=your_api_id_here
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_PHONE=your_phone_number
TARGET_USER_IDS=user_id_1,user_id_2
BUSINESS_KEYWORDS=видеопродакшн,съемка,монтаж,рекламные ролики
```

### 4. Запуск бота
```bash
python main_universal.py
```

## 🌐 Поддерживаемые сферы

### Видеопродакшн
```env
BUSINESS_DOMAIN=video_production
BUSINESS_KEYWORDS=видеопродакшн,съемка,монтаж,рекламные ролики,видеоконтент,цветокоррекция
FULL_CYCLE_PHRASES=полный цикл,под ключ,от концепции до,съемка и монтаж
```

### Веб-разработка
```env
BUSINESS_DOMAIN=web_development
BUSINESS_KEYWORDS=веб-разработка,сайт,приложение,frontend,backend,полный цикл разработки
FULL_CYCLE_PHRASES=полный цикл разработки,под ключ,от дизайна до,разработка и тестирование
```

### Дизайн
```env
BUSINESS_DOMAIN=design
BUSINESS_KEYWORDS=дизайн,логотип,брендинг,графический дизайн,UI/UX,веб-дизайн
FULL_CYCLE_PHRASES=полный цикл дизайна,под ключ,от концепции до,дизайн и разработка
```

### Маркетинг
```env
BUSINESS_DOMAIN=marketing
BUSINESS_KEYWORDS=маркетинг,SMM,реклама,продвижение,контент-маркетинг,digital маркетинг
FULL_CYCLE_PHRASES=полный цикл маркетинга,под ключ,от стратегии до,планирование и реализация
```

### Фотография
```env
BUSINESS_DOMAIN=photography
BUSINESS_KEYWORDS=фотография,фотосессия,свадебная фотография,портретная съемка,обработка фото
FULL_CYCLE_PHRASES=полный цикл фотосессии,под ключ,от съемки до,фото и обработка
```

## 📋 Команды бота

| Команда | Описание |
|---------|----------|
| `/help` | Справка по командам |
| `/stats` | Статистика модели и бота |
| `/train` | Переобучение модели |
| `/correct_<id>` | Отметить сообщение как релевантное |
| `/wrong_<id>` | Отметить сообщение как нерелевантное |
| `/clear_history` | Очистить старую историю |

## 🔧 Настройки

### Машинное обучение
```env
ML_SIMILARITY_THRESHOLD=0.7          # Порог семантического сходства (0-1)
ML_MIN_TRAINING_EXAMPLES=3           # Минимум примеров для обучения
ML_AUTO_TRAIN_THRESHOLD=2            # Частота автообучения
ML_CLASSIFIER_MODEL=production_classifier  # Имя модели
```

### Фильтрация
```env
FILTER_MIN_LENGTH=5                  # Минимальная длина сообщения
FILTER_BLACKLIST=спам,реклама        # Черный список слов
FILTER_FORWARD_PATTERNS=пересланное сообщение,forwarded message  # Паттерны пересылки
```

## 📊 Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram API  │───▶│   Telegram Bot  │───▶│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ ML Classifier   │
                       │ + Utils         │
                       └─────────────────┘
```

### Компоненты

- **`config.py`** - Система конфигурации с переменными окружения
- **`database.py`** - Менеджер SQLite базы данных
- **`ml_classifier.py`** - Классификатор с автообучением
- **`telegram_bot.py`** - Основной класс Telegram бота
- **`utils.py`** - Утилиты для работы с текстом
- **`main_universal.py`** - Главный файл приложения

## 📈 Метрики и статистика

Бот автоматически собирает метрики:

- **Точность модели** - процент правильных предсказаний
- **F1-мера** - баланс точности и полноты
- **Статистика сообщений** - обработано/переслано/отклонено
- **Баланс классов** - соотношение положительных/отрицательных примеров

## 🛠️ Разработка

### Структура проекта

```
├── main_universal.py      # Главный файл
├── config.py              # Конфигурация
├── database.py            # База данных
├── ml_classifier.py       # Машинное обучение
├── telegram_bot.py         # Telegram API
├── utils.py               # Утилиты
├── requirements.txt       # Зависимости
├── env_example.txt        # Пример конфигурации
└── README.md             # Документация
```

### Добавление новой сферы

1. Добавьте конфигурацию в `utils.py` → `get_business_domain_examples()`
2. Обновите примеры в `env_example.txt`
3. Протестируйте с новыми ключевыми словами

## 🔒 Безопасность

- Все чувствительные данные хранятся в переменных окружения
- Сессия Telegram сохраняется локально
- База данных SQLite для приватности данных
- Логирование для мониторинга

## 📝 Лицензия

MIT License - используйте свободно для любых целей.

## 🤝 Поддержка

Если у вас есть вопросы или предложения:

1. Проверьте документацию
2. Посмотрите примеры конфигурации
3. Создайте issue с описанием проблемы

## 🤝 Участие в проекте

Мы приветствуем вклад от сообщества! Пожалуйста, прочитайте [CONTRIBUTING.md](CONTRIBUTING.md) для получения подробной информации.

### Как внести вклад:
1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Создайте Pull Request

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для подробностей.

## 🆘 Поддержка

- 📖 [Документация](README.md)
- 🐛 [Сообщить об ошибке](https://github.com/yourusername/universal-telegram-bot/issues)
- 💡 [Предложить функцию](https://github.com/yourusername/universal-telegram-bot/issues)
- 💬 [Обсуждения](https://github.com/yourusername/universal-telegram-bot/discussions)

## ⭐ Звезды

Если проект вам понравился, поставьте звезду! ⭐

---

**Создано с ❤️ для автоматизации бизнес-процессов**
