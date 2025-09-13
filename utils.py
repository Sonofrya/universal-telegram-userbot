"""
Утилиты для работы с текстом и анализа сообщений
"""
import re
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from config import config

def clean_text(text: str) -> str:
    """Очищает текст от лишних символов"""
    if not text:
        return ""
    
    # Удаляем хештеги и упоминания
    text = re.sub(r'[#@]\w+', '', text)
    
    # Удаляем URL
    text = re.sub(r'http\S+', '', text)
    
    # Удаляем все кроме букв, цифр и пробелов
    text = re.sub(r'[^\w\sа-яА-ЯёЁ]', ' ', text)
    
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def contains_full_cycle_phrases(text: str) -> bool:
    """Проверяет наличие фраз, указывающих на полный цикл"""
    if not text:
        return False
    
    text_lower = text.lower()
    for phrase in config.business.full_cycle_phrases:
        if phrase in text_lower:
            return True
    return False

def is_about_full_cycle_production(text: str) -> bool:
    """Определяет, относится ли сообщение к полному циклу производства"""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Проверяем наличие фраз о полном цикле
    if contains_full_cycle_phrases(text_lower):
        return True
    
    # Проверяем комбинации ключевых слов, указывающих на полный цикл
    # Это универсальная логика, которая работает для любой сферы
    keywords_lower = [kw.lower() for kw in config.business.keywords]
    
    # Ищем слова, связанные с планированием/концепцией
    planning_words = ['концепц', 'идея', 'планирован', 'стратег', 'анализ', 'исследован']
    has_planning = any(word in text_lower for word in planning_words)
    
    # Ищем слова, связанные с производством/реализацией
    production_words = ['производств', 'создан', 'разработк', 'реализац', 'выполнен']
    has_production = any(word in text_lower for word in production_words)
    
    # Ищем слова, связанные с завершением/постобработкой
    completion_words = ['завершен', 'готов', 'финальн', 'итогов', 'результат']
    has_completion = any(word in text_lower for word in completion_words)
    
    # Если есть все три этапа - это полный цикл
    if has_planning and has_production and has_completion:
        return True
    
    # Если есть упоминание нескольких этапов
    stages_count = sum([has_planning, has_production, has_completion])
    if stages_count >= 2 and any(phrase in text_lower for phrase in ['полный', 'комплексный', 'под ключ']):
        return True
    
    return False

def calculate_similarity(model: SentenceTransformer, text: str, keywords: List[str]) -> float:
    """Рассчитывает максимальное сходство текста с ключевыми словами"""
    if not text or not keywords:
        return 0.0
    
    try:
        # Создаем эмбеддинги
        text_embedding = model.encode([text.lower()])[0]
        keyword_embeddings = model.encode([kw.lower() for kw in keywords])
        
        # Рассчитываем сходство
        similarities = [1 - cosine(text_embedding, kw_emb) for kw_emb in keyword_embeddings]
        return max(similarities) if similarities else 0.0
        
    except Exception as e:
        logging.error(f"❌ Ошибка при расчете сходства: {e}")
        return 0.0

def contains_blacklisted_words(text: str) -> bool:
    """Проверяет наличие слов из черного списка"""
    if not text:
        return False
    
    text_lower = text.lower()
    for word in config.filter.blacklist_words:
        if re.search(r'\b' + re.escape(word.lower()) + r'\b', text_lower):
            return True
    return False

def is_forward_notification(text: str) -> bool:
    """Проверяет, является ли текст служебным сообщением о пересылке"""
    if not text:
        return False
    
    text_lower = text.lower()
    for pattern in config.filter.forward_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False

def is_too_short(text: str) -> bool:
    """Проверяет, слишком ли короткое сообщение"""
    if not text:
        return True
    
    words = text.split()
    return len(words) < config.filter.min_message_length

def extract_keywords_from_text(text: str, min_length: int = 3) -> List[str]:
    """Извлекает ключевые слова из текста"""
    if not text:
        return []
    
    # Очищаем текст
    cleaned = clean_text(text)
    
    # Разбиваем на слова
    words = cleaned.split()
    
    # Фильтруем по длине и убираем стоп-слова
    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 'о', 'об', 'за', 'при', 'через'}
    keywords = [word.lower() for word in words 
                if len(word) >= min_length and word.lower() not in stop_words]
    
    return keywords

def calculate_text_complexity(text: str) -> dict:
    """Рассчитывает сложность текста"""
    if not text:
        return {'complexity': 0, 'word_count': 0, 'sentence_count': 0}
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    # Простая метрика сложности
    complexity = (avg_word_length * 0.3 + avg_sentence_length * 0.7) / 10
    
    return {
        'complexity': complexity,
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length
    }

def format_message_info(message_data: dict) -> str:
    """Форматирует информацию о сообщении для отправки"""
    try:
        info_parts = []
        
        if message_data.get('message_date'):
            info_parts.append(f"📅 {message_data['message_date']}")
        
        if message_data.get('sender_info'):
            info_parts.append(f"👤 {message_data['sender_info']}")
        
        if message_data.get('chat_title'):
            info_parts.append(f"💬 {message_data['chat_title']}")
        
        if message_data.get('message_id'):
            info_parts.append(f"🔗 ID: {message_data['message_id']}")
        
        if message_data.get('similarity_score') is not None:
            info_parts.append(f"🎯 Сходство: {message_data['similarity_score']:.3f}")
        
        if message_data.get('ml_probability') is not None:
            info_parts.append(f"🤖 ML: {message_data['ml_probability']:.3f}")
        
        if message_data.get('is_full_cycle') is not None:
            info_parts.append(f"🔁 Полный цикл: {'Да' if message_data['is_full_cycle'] else 'Нет'}")
        
        return '\n'.join(info_parts) + '\n\n'
        
    except Exception as e:
        logging.error(f"❌ Ошибка форматирования информации о сообщении: {e}")
        return "❌ Ошибка форматирования\n\n"

def validate_config() -> List[str]:
    """Проверяет корректность конфигурации"""
    errors = []
    
    if not config.telegram.api_id:
        errors.append("TELEGRAM_API_ID не установлен")
    
    if not config.telegram.api_hash:
        errors.append("TELEGRAM_API_HASH не установлен")
    
    if not config.telegram.phone_number:
        errors.append("TELEGRAM_PHONE не установлен")
    
    if not config.business.keywords:
        errors.append("BUSINESS_KEYWORDS не установлены")
    
    if not config.business.target_user_ids:
        errors.append("TARGET_USER_IDS не установлены")
    
    if config.ml.similarity_threshold < 0 or config.ml.similarity_threshold > 1:
        errors.append("ML_SIMILARITY_THRESHOLD должен быть между 0 и 1")
    
    if config.filter.min_message_length < 1:
        errors.append("FILTER_MIN_LENGTH должен быть больше 0")
    
    return errors

def get_business_domain_examples() -> dict:
    """Возвращает примеры конфигурации для разных сфер"""
    return {
        'video_production': {
            'keywords': 'видеопродакшн,съемка,монтаж,рекламные ролики,видеоконтент,цветокоррекция',
            'full_cycle_phrases': 'полный цикл,под ключ,от концепции до,съемка и монтаж',
            'description': 'Видеопродакшн и создание видеоконтента'
        },
        'web_development': {
            'keywords': 'веб-разработка,сайт,приложение,frontend,backend,полный цикл разработки',
            'full_cycle_phrases': 'полный цикл разработки,под ключ,от дизайна до,разработка и тестирование',
            'description': 'Веб-разработка и создание приложений'
        },
        'design': {
            'keywords': 'дизайн,логотип,брендинг,графический дизайн,UI/UX,веб-дизайн',
            'full_cycle_phrases': 'полный цикл дизайна,под ключ,от концепции до,дизайн и разработка',
            'description': 'Графический дизайн и брендинг'
        },
        'marketing': {
            'keywords': 'маркетинг,SMM,реклама,продвижение,контент-маркетинг,digital маркетинг',
            'full_cycle_phrases': 'полный цикл маркетинга,под ключ,от стратегии до,планирование и реализация',
            'description': 'Маркетинг и продвижение'
        },
        'photography': {
            'keywords': 'фотография,фотосессия,свадебная фотография,портретная съемка,обработка фото',
            'full_cycle_phrases': 'полный цикл фотосессии,под ключ,от съемки до,фото и обработка',
            'description': 'Фотография и фотосессии'
        }
    }

def create_config_template(domain: str) -> str:
    """Создает шаблон конфигурации для указанной сферы"""
    examples = get_business_domain_examples()
    
    if domain not in examples:
        domain = 'general'
        template = {
            'keywords': 'ключевое слово 1,ключевое слово 2,ключевое слово 3',
            'full_cycle_phrases': 'полный цикл,под ключ,комплексный',
            'description': 'Общая сфера деятельности'
        }
    else:
        template = examples[domain]
    
    config_template = f"""# Конфигурация для сферы: {template['description']}
BUSINESS_DOMAIN={domain}
BUSINESS_KEYWORDS={template['keywords']}
FULL_CYCLE_PHRASES={template['full_cycle_phrases']}

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
    
    return config_template
