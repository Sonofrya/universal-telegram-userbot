import logging
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import warnings
import re
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# Отключаем warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s',
    handlers=[logging.FileHandler('userbot.log'), logging.StreamHandler()]
)

# Хранение конфигурации
API_ID = '22794450'
API_HASH = 'eed412e2a2b29fa253407fc60e634e20'
PHONE_NUMBER = '79538703044'

# Ключевые слова для полного цикла продакшена
KEYWORDS_STRING = (
    'полный цикл, видеопродюсерская компания, продакшн полного цикла, '
    'комплексный продакшн, производство видео под ключ, съемка под ключ, '
    'разработка концепции, написание сценария, сценарий, съемка, монтаж, '
    'цветокоррекция, упаковка видео, рекламные рилсы, рекламные reels, '
    'рекламные ролики, видеопродакшн, видеопроизводство, производство видео, '
    'видеоконтент, видеоролики, видео для бренда, видео для бизнеса, рилс, reels, сниппет'
)
KEYWORDS = [kw.strip().lower() for kw in KEYWORDS_STRING.split(', ')]
TARGET_USER_IDS = ['7013831967', '406521857']
SIMILARITY_THRESHOLD = 0.7
SESSION_FILE = 'session.txt'

# Черный список
BLACKLIST_WORDS = [
    'фриланс', 'фрилансер'
]

# Фразы полного цикла
FULL_CYCLE_PHRASES = [
    'полный цикл', 'под ключ', 'продакшн полного цикла', 'комплексный', 'от и до', 
    'от концепции до', 'от съемки до', 'от идеи до',
    'разработка концепции', 'написание сценария', 'съемка и монтаж',
    'монтаж и цветокоррекция', 'цветокоррекция и упаковка',
    'весь цикл производства', 'полное производство'
]

FORWARD_PATTERNS = [
    r'пересланное сообщение', r'forwarded message', r'было переслано',
    r'forwarded from', r'сообщение переслано', r'медиа переслано'
]

processed_messages = set()
feedback_db = {}

def clean_text(text):
    """Очищает текст от лишних символов"""
    if not text:
        return ""
    
    text = re.sub(r'[#@]\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\sа-яА-ЯёЁ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def contains_full_cycle_phrases(text):
    """Проверяет наличие фраз, указывающих на полный цикл"""
    if not text:
        return False
    
    text_lower = text.lower()
    for phrase in FULL_CYCLE_PHRASES:
        if phrase in text_lower:
            return True
    return False

def is_about_full_cycle_production(text):
    """Определяет, относится ли сообщение к полному циклу производства"""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Проверяем наличие фраз о полном цикле
    if contains_full_cycle_phrases(text_lower):
        return True
    
    # Проверяем комбинации ключевых слов, указывающих на полный цикл
    has_concept = any(word in text_lower for word in ['концепц', 'идея', 'сценарий'])
    has_production = any(word in text_lower for word in ['съемк', 'производств', 'монтаж'])
    has_post = any(word in text_lower for word in ['цветокоррекц', 'упаковк', 'график'])
    
    # Если есть все три этапа - это полный цикл
    if has_concept and has_production and has_post:
        return True
    
    # Если есть упоминание нескольких этапов
    stages_count = sum([has_concept, has_production, has_post])
    if stages_count >= 2 and 'полный' in text_lower:
        return True
    
    return False

def calculate_similarity(model, text, keyword_embeddings):
    """Рассчитывает максимальное сходство текста с ключевыми словами"""
    if not text:
        return 0.0
    
    try:
        text_embedding = model.encode([text.lower()])[0]
        similarities = [1 - cosine(text_embedding, kw_emb) for kw_emb in keyword_embeddings]
        return max(similarities) if similarities else 0.0
    except Exception as e:
        logging.error(f"Ошибка при расчете сходства: {e}")
        return 0.0

def contains_blacklisted_words(text):
    """Проверяет наличие слов из черного списка"""
    if not text:
        return False
    
    text_lower = text.lower()
    for word in BLACKLIST_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return True
    return False

def is_forward_notification(text):
    """Проверяет, является ли текст служебным сообщением о пересылке"""
    if not text:
        return False
    
    text_lower = text.lower()
    for pattern in FORWARD_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False

def is_too_short(text):
    """Пропускаем слишком короткие сообщения"""
    if not text:
        return True
    words = text.split()
    return len(words) < 5

def get_sender_info(sender):
    """Получает полную информацию об отправителе"""
    if not sender:
        return "Неизвестный отправитель"
    
    info_parts = []
    
    if hasattr(sender, 'first_name') and sender.first_name:
        info_parts.append(sender.first_name)
    if hasattr(sender, 'last_name') and sender.last_name:
        info_parts.append(sender.last_name)
    if hasattr(sender, 'username') and sender.username:
        info_parts.append(f"(@{sender.username})")
    if hasattr(sender, 'title') and sender.title:
        info_parts.append(sender.title)
    
    return ' '.join(info_parts) if info_parts else "Неизвестный отправитель"

async def get_entity(client, user_id):
    try:
        # Пробуем как число (если это цифровой ID)
        if isinstance(user_id, str) and user_id.isdigit():
            user_entity = await client.get_entity(int(user_id))
        else:
            # Пробуем как username/номер телефона
            user_entity = await client.get_entity(str(user_id))
        return user_entity
    except Exception as e:
        logging.error(f"Ошибка при получении сущности для пользователя {user_id}: {e}")
        return None

async def copy_message_content(client, message, target_user):
    """Копирует содержимое сообщения вместо пересылки."""
    try:
        sent_message = None
        
        if message.text:
            sent_message = await client.send_message(target_user, message.text)
        
        if message.media and not isinstance(message.media, type(None)):
            if sent_message:
                await client.send_file(target_user, message.media, reply_to=sent_message.id)
            else:
                sent_message = await client.send_file(target_user, message.media)
        
        return sent_message
        
    except Exception as e:
        logging.error(f"Ошибка при копировании сообщения: {e}")
        return None

class MessageClassifier:
    def __init__(self, model_name='production_classifier'):
        self.model_name = model_name
        self.classifier = None
        self.training_data = []
        self.is_trained = False
        self.last_training_size = 0
        self.load_model()
    
    def load_model(self):
        """Загружаем модель и данные обучения с диска"""
        try:
            # Загружаем модель
            if os.path.exists(f'{self.model_name}.pkl'):
                with open(f'{self.model_name}.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.classifier = data['classifier']
                    self.is_trained = True
                    logging.info("✓ Модель классификатора загружена")
            
            # Загружаем данные обучения
            if os.path.exists(f'{self.model_name}_data.pkl'):
                with open(f'{self.model_name}_data.pkl', 'rb') as f:
                    self.training_data = pickle.load(f)
                    self.last_training_size = len(self.training_data)
                    logging.info(f"✓ Загружено {len(self.training_data)} примеров для обучения")
                    
        except Exception as e:
            logging.error(f"✗ Ошибка загрузки модели: {e}")
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
            self.training_data = []
            self.is_trained = False
    
    def save_model(self):
        """Сохраняем модель и данные обучения на диск"""
        try:
            # Сохраняем обученную модель
            if self.classifier and self.is_trained:
                with open(f'{self.model_name}.pkl', 'wb') as f:
                    pickle.dump({
                        'classifier': self.classifier,
                        'saved_at': datetime.now(),
                        'training_size': len(self.training_data)
                    }, f)
            
            # Всегда сохраняем данные обучения
            with open(f'{self.model_name}_data.pkl', 'wb') as f:
                pickle.dump(self.training_data, f)
                
            logging.info("✓ Модель и данные сохранены на диск")
            
        except Exception as e:
            logging.error(f"✗ Ошибка сохранения модели: {e}")
    
    def add_training_example(self, text, label, model):
        """Добавляем пример для обучения и автоматически обучаем при необходимости"""
        try:
            # Очищаем текст и получаем эмбеддинг
            cleaned_text = clean_text(text)
            embedding = model.encode([cleaned_text])[0]
            
            # Сохраняем данные
            self.training_data.append({
                'text': text,
                'embedding': embedding,
                'label': label,
                'added_at': datetime.now()
            })
            
            # Сохраняем на диск
            self.save_model()
            
            logging.info(f"✓ Добавлен пример обучения (всего: {len(self.training_data)})")
            
            # Автоматическое обучение при накоплении достаточного количества примеров
            if len(self.training_data) >= 3 and len(self.training_data) % 2 == 0:
                self.auto_train(model)
            
        except Exception as e:
            logging.error(f"✗ Ошибка добавления примера: {e}")
    
    def auto_train(self, model):
        """Автоматическое обучение при накоплении примеров"""
        if len(self.training_data) < 3:
            return False
        
        try:
            # Создаем новую модель если нужно
            if self.classifier is None:
                self.classifier = LogisticRegression(random_state=42, max_iter=1000)
            
            # Подготавливаем данные
            X = np.array([item['embedding'] for item in self.training_data])
            y = np.array([item['label'] for item in self.training_data])
            
            # Обучаем модель
            self.classifier.fit(X, y)
            self.is_trained = True
            
            # Сохраняем обновленную модель
            self.save_model()
            
            accuracy = self.classifier.score(X, y)
            self.last_training_size = len(self.training_data)
            
            logging.info(f"✅ Модель автоматически переобучена!")
            logging.info(f"📊 На {len(self.training_data)} примерах, точность: {accuracy:.2f}")
            
            return True
            
        except Exception as e:
            logging.error(f"✗ Ошибка автоматического обучения: {e}")
            return False
    
    def train(self, model):
        """Ручное обучение модели"""
        return self.auto_train(model)
    
    def predict(self, text, model):
        """Предсказываем вероятность для текста"""
        if not self.is_trained or self.classifier is None:
            return None
        
        try:
            cleaned_text = clean_text(text)
            embedding = model.encode([cleaned_text])[0]
            probability = self.classifier.predict_proba([embedding])[0][1]
            return probability
            
        except Exception as e:
            logging.error(f"✗ Ошибка предсказания: {e}")
            return None
    
    def get_stats(self):
        """Получаем статистику модели"""
        if not self.is_trained or len(self.training_data) == 0:
            return {
                'is_trained': False,
                'training_examples': len(self.training_data),
                'accuracy': None
            }
        
        try:
            X = np.array([item['embedding'] for item in self.training_data])
            y = np.array([item['label'] for item in self.training_data])
            accuracy = self.classifier.score(X, y)
            
            return {
                'is_trained': True,
                'training_examples': len(self.training_data),
                'accuracy': accuracy,
                'last_training_size': self.last_training_size
            }
        except:
            return {
                'is_trained': False,
                'training_examples': len(self.training_data),
                'accuracy': None
            }

async def forward_messages(event, model, keyword_embeddings, client, classifier):
    """Пересылка сообщений только о полном цикле производства"""
    
    # Пропускаем свои сообщения и уже обработанные
    if event.message.out or event.message.id in processed_messages:
        return
    
    message_text = event.message.text if event.message.text else ""
    cleaned_text = clean_text(message_text)
    
    # Пропускаем служебные сообщения о пересылке
    if is_forward_notification(message_text):
        logging.info("Пропущено служебное сообщение о пересылке")
        return
    
    # Пропускаем слишком короткие сообщения
    if is_too_short(cleaned_text):
        logging.info(f"Пропущено короткое сообщение: '{cleaned_text}'")
        return
    
    # Пропускаем сообщения из черного списка
    if contains_blacklisted_words(message_text):
        logging.info(f"Пропущено сообщение из черного списка: '{cleaned_text}'")
        return
    
    processed_messages.add(event.message.id)

    logging.info(f"✗ Сообщение не переслано [ID: {event.message.id}]")
    
    try:
        # Проверяем, относится ли сообщение к полному циклу производства
        is_full_cycle = is_about_full_cycle_production(message_text)
        
        # Рассчитываем семантическое сходство
        max_similarity = calculate_similarity(model, cleaned_text, keyword_embeddings)
        
        # Используем ML модель если она обучена
        ml_probability = classifier.predict(message_text, model)
        use_ml = ml_probability is not None and classifier.is_trained
        
        # Решаем, пересылать ли сообщение
        if use_ml:
            should_forward = ml_probability > 0.5
            logging.info(f"ML модель: {ml_probability:.3f}, Решение: {'✓' if should_forward else '✗'}")
        else:
            should_forward = is_full_cycle or max_similarity > SIMILARITY_THRESHOLD
        
        logging.info(f"Сообщение: '{cleaned_text[:80]}...'")
        logging.info(f"Сходство: {max_similarity:.3f}, Полный цикл: {is_full_cycle}, ML: {ml_probability}")
        
        if should_forward:
            chat_title = event.message.chat.title if hasattr(event.message.chat, 'title') else "Приватный чат"
            
            sender_info = "Неизвестный отправитель"
            try:
                sender = await event.message.get_sender()
                sender_info = get_sender_info(sender)
            except Exception as e:
                logging.warning(f"Не удалось получить информацию об отправителе: {e}")
            
            message_date = event.message.date.strftime("%d.%m.%Y %H:%M") if hasattr(event.message, 'date') else ""
            ml_info = f", ML: {ml_probability:.3f}" if use_ml else ""
            
            message_info = (
                f"📅 {message_date}\n👤 {sender_info}\n💬 {chat_title}\n"
                f"🔗 ID: {event.message.id}\n🎯 Сходство: {max_similarity:.3f}{ml_info}\n"
                f"🔁 Полный цикл: {'Да' if is_full_cycle else 'Нет'}\n\n"
            )
            
            # Сохраняем для обратной связи
            feedback_db[event.message.id] = {
                'text': message_text, 
                'forwarded': True, 
                'timestamp': datetime.now(),
                'similarity': max_similarity,
                'is_full_cycle': is_full_cycle,
                'ml_probability': ml_probability
            }
            
            for user_id in TARGET_USER_IDS:
                try:
                    user_entity = await get_entity(client, user_id)
                    if not user_entity:
                        continue
                    
                    try:
                        forward_message = await client.forward_messages(user_entity, event.message)
                        if forward_message:
                            await client.send_message(user_entity, message_info, reply_to=forward_message.id)
                            logging.info(f"✓ Сообщение переслано пользователю {user_id}")
                            continue
                    except Exception as forward_error:
                        logging.warning(f"Не удалось переслать: {forward_error}")
                    
                    copied_message = await copy_message_content(client, event.message, user_entity)
                    if copied_message:
                        await client.send_message(user_entity, message_info, reply_to=copied_message.id)
                        logging.info(f"✓ Содержимое скопировано пользователю {user_id}")
                    else:
                        logging.error(f"✗ Не удалось скопировать пользователю {user_id}")
                        
                except Exception as e:
                    logging.error(f"✗ Ошибка для пользователя {user_id}: {e}")
        else:
            # Сохраняем отрицательный пример для обучения
            feedback_db[event.message.id] = {
                'text': message_text, 
                'forwarded': False, 
                'timestamp': datetime.now(),
                'similarity': max_similarity,
                'is_full_cycle': is_full_cycle,
                'ml_probability': ml_probability
            }
            logging.info(f"✗ Сообщение не переслано [ID: {event.message.id}]")
                    
    except Exception as e:
        logging.error(f"Ошибка при анализе сообщения: {e}")

async def main():
    # Загрузка сессии если существует
    session_string = None
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, 'r') as f:
            session_string = f.read().strip()
    
    if session_string:
        client = TelegramClient(StringSession(session_string), API_ID, API_HASH)
    else:
        client = TelegramClient(StringSession(), API_ID, API_HASH)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        keyword_embeddings = model.encode(KEYWORDS)
    
    # Инициализируем классификатор с автообучением
    classifier = MessageClassifier()
    stats = classifier.get_stats()
    logging.info(f"Модель загружена: {'обучена' if stats['is_trained'] else 'не обучена'}")
    logging.info(f"Примеров для обучения: {stats['training_examples']}")
    if stats['accuracy']:
        logging.info(f"Точность модели: {stats['accuracy']:.2%}")
    
    # Команды для управления
    @client.on(events.NewMessage(pattern='/train'))
    async def train_handler(event):
        if len(classifier.training_data) >= 3:
            success = classifier.train(model)
            await event.reply("✅ Модель успешно переобучена!" if success else "❌ Ошибка переобучения")
        else:
            await event.reply(f"❌ Недостаточно данных. Нужно 3+ примеров (сейчас: {len(classifier.training_data)})")
    
    @client.on(events.NewMessage(pattern='/stats'))
    async def stats_handler(event):
        stats = classifier.get_stats()
        accuracy_text = f"{stats['accuracy']:.2%}" if stats['accuracy'] else "N/A"
        stats_text = (
            f"📊 Статистика модели:\n"
            f"• Обучена: {'✅' if stats['is_trained'] else '❌'}\n"
            f"• Примеров: {stats['training_examples']}\n"
            f"• Точность: {accuracy_text}\n"
            f"• Сообщений в истории: {len(feedback_db)}"
        )
        await event.reply(stats_text)
    
    @client.on(events.NewMessage(pattern=r'/correct_(\d+)'))
    async def correct_handler(event):
        try:
            msg_id = int(event.pattern_match.group(1))
            if msg_id in feedback_db:
                classifier.add_training_example(feedback_db[msg_id]['text'], 1, model)
                await event.reply("✅ Добавлен положительный пример обучения!")
            else:
                await event.reply("❌ Сообщение не найдено в истории")
        except:
            await event.reply("❌ Используйте: /correct_12345")
    
    @client.on(events.NewMessage(pattern=r'/wrong_(\d+)'))
    async def wrong_handler(event):
        try:
            msg_id = int(event.pattern_match.group(1))
            if msg_id in feedback_db:
                classifier.add_training_example(feedback_db[msg_id]['text'], 0, model)
                await event.reply("✅ Добавлен отрицательный пример обучения!")
            else:
                await event.reply("❌ Сообщение не найдено в истории")
        except:
            await event.reply("❌ Используйте: /wrong_12345")
    
    @client.on(events.NewMessage(pattern='/clear_history'))
    async def clear_history_handler(event):
        feedback_db.clear()
        await event.reply("✅ История сообщений очищена!")
    
    # Основной обработчик сообщений
    @client.on(events.NewMessage)
    async def handler(event):
        await forward_messages(event, model, keyword_embeddings, client, classifier)

    logging.info("Юзербот запущен с автообучением!")
    logging.info("Доступные команды: /train, /stats, /correct_12345, /wrong_12345, /clear_history")
    
    # Сохраняем сессию при первом запуске
    if not os.path.exists(SESSION_FILE):
        await client.start(phone=PHONE_NUMBER)
        with open(SESSION_FILE, 'w') as f:
            f.write(client.session.save())
        logging.info("Сессия сохранена в файл")
    else:
        await client.start()
    
    # 🔧 ДОБАВЛЕНО: Предварительная загрузка сущностей пользователей
    try:
        logging.info("🔍 Предварительная загрузка сущностей пользователей...")
        for user_id in TARGET_USER_IDS:
            try:
                if str(user_id).isdigit():
                    await client.get_entity(int(user_id))
                    logging.info(f"✅ Сущность пользователя {user_id} загружена")
                else:
                    await client.get_entity(str(user_id))
                    logging.info(f"✅ Сущность пользователя {user_id} загружена")
            except Exception as e:
                logging.warning(f"⚠️ Не удалось загрузить сущность для {user_id}: {e}")
                logging.info("ℹ️ Попробуйте отправить сообщение этому пользователю от имени бота")
    except Exception as e:
        logging.warning(f"⚠️ Ошибка при предварительной загрузке сущностей: {e}")
    
    await client.run_until_disconnected()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())