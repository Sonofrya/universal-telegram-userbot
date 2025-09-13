"""
Модуль для работы с Telegram API
"""
import logging
import os
from typing import List, Optional, Dict, Any
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import User, Chat, Channel
from config import config
from database import DatabaseManager
from ml_classifier import UniversalMessageClassifier

class TelegramBot:
    """Основной класс Telegram бота"""
    
    def __init__(self, db_manager: DatabaseManager = None, classifier: UniversalMessageClassifier = None):
        self.db_manager = db_manager or DatabaseManager()
        self.classifier = classifier or UniversalMessageClassifier(db_manager=self.db_manager)
        self.client = None
        self.processed_messages = set()
        self.daily_stats = {
            'processed': 0,
            'forwarded': 0,
            'rejected': 0,
            'training_examples': 0
        }
        
        # Инициализируем клиент
        self._init_client()
    
    def _init_client(self):
        """Инициализирует Telegram клиент"""
        try:
            # Загружаем сессию если существует
            session_string = None
            if os.path.exists(config.telegram.session_file):
                with open(config.telegram.session_file, 'r') as f:
                    session_string = f.read().strip()
            
            if session_string:
                self.client = TelegramClient(StringSession(session_string), config.telegram.api_id, config.telegram.api_hash)
            else:
                self.client = TelegramClient(StringSession(), config.telegram.api_id, config.telegram.api_hash)
                
            logging.info("✅ Telegram клиент инициализирован")
            
        except Exception as e:
            logging.error(f"❌ Ошибка инициализации Telegram клиента: {e}")
            self.client = None
    
    async def start(self):
        """Запускает бота"""
        if not self.client:
            logging.error("❌ Telegram клиент не инициализирован")
            return False
        
        try:
            # Сохраняем сессию при первом запуске
            if not os.path.exists(config.telegram.session_file):
                await self.client.start(phone=config.telegram.phone_number)
                with open(config.telegram.session_file, 'w') as f:
                    f.write(self.client.session.save())
                logging.info("✅ Сессия сохранена")
            else:
                await self.client.start()
            
            # Предварительная загрузка сущностей пользователей
            await self._preload_user_entities()
            
            # Регистрируем обработчики
            self._register_handlers()
            
            logging.info("🚀 Бот запущен!")
            logging.info(f"📊 Статистика модели: {self.classifier.get_stats()}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Ошибка запуска бота: {e}")
            return False
    
    async def _preload_user_entities(self):
        """Предварительно загружает сущности пользователей"""
        try:
            logging.info("🔍 Предварительная загрузка сущностей пользователей...")
            for user_id in config.business.target_user_ids:
                try:
                    entity = await self._get_entity(user_id)
                    if entity:
                        logging.info(f"✅ Сущность пользователя {user_id} загружена")
                    else:
                        logging.warning(f"⚠️ Не удалось загрузить сущность для {user_id}")
                except Exception as e:
                    logging.warning(f"⚠️ Ошибка загрузки сущности {user_id}: {e}")
        except Exception as e:
            logging.warning(f"⚠️ Ошибка предварительной загрузки: {e}")
    
    async def _get_entity(self, user_id: str):
        """Получает сущность пользователя"""
        try:
            if user_id.isdigit():
                return await self.client.get_entity(int(user_id))
            else:
                return await self.client.get_entity(user_id)
        except Exception as e:
            logging.error(f"❌ Ошибка получения сущности {user_id}: {e}")
            return None
    
    def _register_handlers(self):
        """Регистрирует обработчики событий"""
        
        # Команды управления
        @self.client.on(events.NewMessage(pattern='/train'))
        async def train_handler(event):
            await self._handle_train_command(event)
        
        @self.client.on(events.NewMessage(pattern='/stats'))
        async def stats_handler(event):
            await self._handle_stats_command(event)
        
        @self.client.on(events.NewMessage(pattern=r'/correct_(\d+)'))
        async def correct_handler(event):
            await self._handle_correct_command(event)
        
        @self.client.on(events.NewMessage(pattern=r'/wrong_(\d+)'))
        async def wrong_handler(event):
            await self._handle_wrong_command(event)
        
        @self.client.on(events.NewMessage(pattern='/clear_history'))
        async def clear_history_handler(event):
            await self._handle_clear_history_command(event)
        
        @self.client.on(events.NewMessage(pattern='/help'))
        async def help_handler(event):
            await self._handle_help_command(event)
        
        # Основной обработчик сообщений
        @self.client.on(events.NewMessage)
        async def message_handler(event):
            await self._handle_message(event)
    
    async def _handle_train_command(self, event):
        """Обработчик команды /train"""
        try:
            if len(self.classifier.training_data) >= config.ml.min_training_examples:
                success = self.classifier.retrain()
                if success:
                    stats = self.classifier.get_stats()
                    response = (
                        f"✅ Модель успешно переобучена!\n"
                        f"📊 Точность: {stats.get('accuracy', 0):.2%}\n"
                        f"📈 F1-мера: {stats.get('f1_score', 0):.2%}\n"
                        f"📚 Примеров: {stats['training_examples']}"
                    )
                else:
                    response = "❌ Ошибка переобучения модели"
            else:
                response = f"❌ Недостаточно данных. Нужно {config.ml.min_training_examples}+ примеров (сейчас: {len(self.classifier.training_data)})"
            
            await event.reply(response)
            
        except Exception as e:
            logging.error(f"❌ Ошибка обработки команды /train: {e}")
            await event.reply("❌ Ошибка выполнения команды")
    
    async def _handle_stats_command(self, event):
        """Обработчик команды /stats"""
        try:
            # Статистика модели
            ml_stats = self.classifier.get_stats()
            training_stats = self.classifier.get_training_data_stats()
            
            # Статистика бота
            bot_stats = self.db_manager.get_stats_summary(7)
            
            response = (
                f"📊 **Статистика модели:**\n"
                f"• Обучена: {'✅' if ml_stats['is_trained'] else '❌'}\n"
                f"• Примеров: {ml_stats['training_examples']}\n"
                f"• Точность: {ml_stats.get('accuracy', 0):.2%}\n"
                f"• F1-мера: {ml_stats.get('f1_score', 0):.2%}\n\n"
                f"📈 **Данные обучения:**\n"
                f"• Положительных: {training_stats['positive']}\n"
                f"• Отрицательных: {training_stats['negative']}\n"
                f"• Баланс: {training_stats['balance']:.2%}\n\n"
                f"🤖 **Статистика бота (7 дней):**\n"
                f"• Обработано: {bot_stats.get('total_processed', 0)}\n"
                f"• Переслано: {bot_stats.get('total_forwarded', 0)}\n"
                f"• Отклонено: {bot_stats.get('total_rejected', 0)}\n"
                f"• Процент пересылки: {bot_stats.get('forward_rate', 0):.1%}"
            )
            
            await event.reply(response)
            
        except Exception as e:
            logging.error(f"❌ Ошибка обработки команды /stats: {e}")
            await event.reply("❌ Ошибка получения статистики")
    
    async def _handle_correct_command(self, event):
        """Обработчик команды /correct_<id>"""
        try:
            msg_id = int(event.pattern_match.group(1))
            message_data = self.db_manager.get_message(msg_id)
            
            if message_data:
                success = self.classifier.add_training_example(message_data['text'], 1)
                if success:
                    self.daily_stats['training_examples'] += 1
                    await event.reply("✅ Добавлен положительный пример обучения!")
                else:
                    await event.reply("❌ Ошибка добавления примера")
            else:
                await event.reply("❌ Сообщение не найдено в истории")
                
        except Exception as e:
            logging.error(f"❌ Ошибка обработки команды /correct: {e}")
            await event.reply("❌ Используйте: /correct_12345")
    
    async def _handle_wrong_command(self, event):
        """Обработчик команды /wrong_<id>"""
        try:
            msg_id = int(event.pattern_match.group(1))
            message_data = self.db_manager.get_message(msg_id)
            
            if message_data:
                success = self.classifier.add_training_example(message_data['text'], 0)
                if success:
                    self.daily_stats['training_examples'] += 1
                    await event.reply("✅ Добавлен отрицательный пример обучения!")
                else:
                    await event.reply("❌ Ошибка добавления примера")
            else:
                await event.reply("❌ Сообщение не найдено в истории")
                
        except Exception as e:
            logging.error(f"❌ Ошибка обработки команды /wrong: {e}")
            await event.reply("❌ Используйте: /wrong_12345")
    
    async def _handle_clear_history_command(self, event):
        """Обработчик команды /clear_history"""
        try:
            # Очищаем старые данные (старше 30 дней)
            success = self.db_manager.clear_old_data(30)
            if success:
                await event.reply("✅ История сообщений очищена!")
            else:
                await event.reply("❌ Ошибка очистки истории")
                
        except Exception as e:
            logging.error(f"❌ Ошибка обработки команды /clear_history: {e}")
            await event.reply("❌ Ошибка очистки истории")
    
    async def _handle_help_command(self, event):
        """Обработчик команды /help"""
        help_text = (
            f"🤖 **Универсальный Telegram-бот**\n"
            f"Домен: {config.business.business_domain}\n\n"
            f"📋 **Доступные команды:**\n"
            f"• `/stats` - статистика модели и бота\n"
            f"• `/train` - переобучение модели\n"
            f"• `/correct_<id>` - отметить сообщение как релевантное\n"
            f"• `/wrong_<id>` - отметить сообщение как нерелевантное\n"
            f"• `/clear_history` - очистить старую историю\n"
            f"• `/help` - эта справка\n\n"
            f"🔍 **Ключевые слова:** {', '.join(config.business.keywords[:5])}...\n"
            f"🎯 **Порог сходства:** {config.ml.similarity_threshold}\n"
            f"📚 **Примеров для обучения:** {len(self.classifier.training_data)}"
        )
        
        await event.reply(help_text)
    
    async def _handle_message(self, event):
        """Основной обработчик сообщений"""
        try:
            await self._process_message(event)
        except Exception as e:
            logging.error(f"❌ Ошибка обработки сообщения: {e}")
    
    async def _process_message(self, event):
        """Обрабатывает входящее сообщение"""
        # Пропускаем свои сообщения и уже обработанные
        if event.message.out or event.message.id in self.processed_messages:
            return
        
        self.processed_messages.add(event.message.id)
        self.daily_stats['processed'] += 1
        
        message_text = event.message.text or ""
        
        # Проверяем фильтры
        if not self._passes_filters(message_text):
            self.daily_stats['rejected'] += 1
            return
        
        # Анализируем сообщение
        analysis = await self._analyze_message(message_text)
        
        # Сохраняем в базу данных
        message_data = {
            'message_id': event.message.id,
            'text': message_text,
            'sender_info': await self._get_sender_info(event),
            'chat_title': self._get_chat_title(event),
            'message_date': event.message.date.strftime("%d.%m.%Y %H:%M") if event.message.date else "",
            'similarity_score': analysis['similarity'],
            'is_full_cycle': analysis['is_full_cycle'],
            'ml_probability': analysis['ml_probability'],
            'forwarded': analysis['should_forward']
        }
        
        self.db_manager.save_message(message_data)
        
        # Пересылаем если нужно
        if analysis['should_forward']:
            await self._forward_message(event, analysis, message_data)
            self.daily_stats['forwarded'] += 1
        else:
            self.daily_stats['rejected'] += 1
            logging.info(f"✗ Сообщение не переслано [ID: {event.message.id}]")
    
    def _passes_filters(self, text: str) -> bool:
        """Проверяет, проходит ли сообщение фильтры"""
        if not text:
            return False
        
        # Проверяем минимальную длину
        if len(text.split()) < config.filter.min_message_length:
            logging.info(f"Пропущено короткое сообщение: '{text[:50]}...'")
            return False
        
        # Проверяем черный список
        text_lower = text.lower()
        for word in config.filter.blacklist_words:
            if word.lower() in text_lower:
                logging.info(f"Пропущено сообщение из черного списка: '{text[:50]}...'")
                return False
        
        # Проверяем служебные сообщения о пересылке
        for pattern in config.filter.forward_patterns:
            import re
            if re.search(pattern, text_lower, re.IGNORECASE):
                logging.info("Пропущено служебное сообщение о пересылке")
                return False
        
        return True
    
    async def _analyze_message(self, text: str) -> Dict[str, Any]:
        """Анализирует сообщение на релевантность"""
        from utils import clean_text, calculate_similarity, is_about_full_cycle_production
        
        cleaned_text = clean_text(text)
        
        # Семантическое сходство
        similarity = calculate_similarity(
            self.classifier.sentence_model, 
            cleaned_text, 
            config.business.keywords
        )
        
        # Проверка на полный цикл
        is_full_cycle = is_about_full_cycle_production(text)
        
        # ML предсказание
        ml_probability = self.classifier.predict(text)
        
        # Решение о пересылке
        if ml_probability is not None and self.classifier.is_trained:
            should_forward = ml_probability > 0.5
        else:
            should_forward = is_full_cycle or similarity > config.ml.similarity_threshold
        
        return {
            'similarity': similarity,
            'is_full_cycle': is_full_cycle,
            'ml_probability': ml_probability,
            'should_forward': should_forward
        }
    
    async def _get_sender_info(self, event) -> str:
        """Получает информацию об отправителе"""
        try:
            sender = await event.message.get_sender()
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
            
        except Exception as e:
            logging.warning(f"Не удалось получить информацию об отправителе: {e}")
            return "Неизвестный отправитель"
    
    def _get_chat_title(self, event) -> str:
        """Получает название чата"""
        try:
            if hasattr(event.message.chat, 'title') and event.message.chat.title:
                return event.message.chat.title
            return "Приватный чат"
        except:
            return "Неизвестный чат"
    
    async def _forward_message(self, event, analysis: Dict[str, Any], message_data: Dict[str, Any]):
        """Пересылает сообщение целевым пользователям"""
        try:
            chat_title = message_data['chat_title']
            sender_info = message_data['sender_info']
            message_date = message_data['message_date']
            
            ml_info = f", ML: {analysis['ml_probability']:.3f}" if analysis['ml_probability'] is not None else ""
            
            message_info = (
                f"📅 {message_date}\n"
                f"👤 {sender_info}\n"
                f"💬 {chat_title}\n"
                f"🔗 ID: {event.message.id}\n"
                f"🎯 Сходство: {analysis['similarity']:.3f}{ml_info}\n"
                f"🔁 Полный цикл: {'Да' if analysis['is_full_cycle'] else 'Нет'}\n\n"
            )
            
            for user_id in config.business.target_user_ids:
                try:
                    user_entity = await self._get_entity(user_id)
                    if not user_entity:
                        continue
                    
                    # Пробуем переслать
                    try:
                        forward_message = await self.client.forward_messages(user_entity, event.message)
                        if forward_message:
                            await self.client.send_message(user_entity, message_info, reply_to=forward_message.id)
                            logging.info(f"✅ Сообщение переслано пользователю {user_id}")
                            continue
                    except Exception as forward_error:
                        logging.warning(f"Не удалось переслать: {forward_error}")
                    
                    # Если не получилось переслать, копируем содержимое
                    await self._copy_message_content(event, user_entity, message_info)
                    
                except Exception as e:
                    logging.error(f"❌ Ошибка для пользователя {user_id}: {e}")
                    
        except Exception as e:
            logging.error(f"❌ Ошибка пересылки сообщения: {e}")
    
    async def _copy_message_content(self, event, target_user, message_info: str):
        """Копирует содержимое сообщения"""
        try:
            sent_message = None
            
            if event.message.text:
                sent_message = await self.client.send_message(target_user, event.message.text)
            
            if event.message.media and not isinstance(event.message.media, type(None)):
                if sent_message:
                    await self.client.send_file(target_user, event.message.media, reply_to=sent_message.id)
                else:
                    sent_message = await self.client.send_file(target_user, event.message.media)
            
            if sent_message:
                await self.client.send_message(target_user, message_info, reply_to=sent_message.id)
                logging.info(f"✅ Содержимое скопировано пользователю")
            else:
                logging.error("❌ Не удалось скопировать содержимое")
                
        except Exception as e:
            logging.error(f"❌ Ошибка копирования содержимого: {e}")
    
    async def run(self):
        """Запускает бота и держит его работающим"""
        if await self.start():
            await self.client.run_until_disconnected()
        else:
            logging.error("❌ Не удалось запустить бота")
    
    async def stop(self):
        """Останавливает бота"""
        if self.client:
            await self.client.disconnect()
            logging.info("🛑 Бот остановлен")
