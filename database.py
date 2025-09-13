"""
Модуль для работы с базой данных
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import numpy as np

class DatabaseManager:
    """Менеджер базы данных для хранения данных бота"""
    
    def __init__(self, db_path: str = 'bot_database.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных и создание таблиц"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица для сообщений
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    sender_info TEXT,
                    chat_title TEXT,
                    message_date TEXT,
                    similarity_score REAL,
                    is_full_cycle BOOLEAN,
                    ml_probability REAL,
                    forwarded BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица для обучения классификатора
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    label INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица для метрик модели
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    training_examples INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица для статистики бота
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    messages_processed INTEGER DEFAULT 0,
                    messages_forwarded INTEGER DEFAULT 0,
                    messages_rejected INTEGER DEFAULT 0,
                    training_examples_added INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logging.info("✅ База данных инициализирована")
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для работы с подключением к БД"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_message(self, message_data: Dict[str, Any]) -> bool:
        """Сохраняет сообщение в базу данных"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO messages 
                    (message_id, text, sender_info, chat_title, message_date, 
                     similarity_score, is_full_cycle, ml_probability, forwarded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message_data['message_id'],
                    message_data['text'],
                    message_data.get('sender_info', ''),
                    message_data.get('chat_title', ''),
                    message_data.get('message_date', ''),
                    message_data.get('similarity_score', 0.0),
                    message_data.get('is_full_cycle', False),
                    message_data.get('ml_probability', 0.0),
                    message_data.get('forwarded', False)
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"❌ Ошибка сохранения сообщения: {e}")
            return False
    
    def get_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Получает сообщение по ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM messages WHERE message_id = ?', (message_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logging.error(f"❌ Ошибка получения сообщения: {e}")
            return None
    
    def save_training_example(self, text: str, embedding: np.ndarray, label: int) -> bool:
        """Сохраняет пример для обучения"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Конвертируем numpy array в bytes
                embedding_bytes = embedding.tobytes()
                cursor.execute('''
                    INSERT INTO training_data (text, embedding, label)
                    VALUES (?, ?, ?)
                ''', (text, embedding_bytes, label))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"❌ Ошибка сохранения примера обучения: {e}")
            return False
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Получает все данные для обучения"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM training_data ORDER BY created_at DESC')
                rows = cursor.fetchall()
                training_data = []
                for row in rows:
                    data = dict(row)
                    # Конвертируем bytes обратно в numpy array
                    data['embedding'] = np.frombuffer(data['embedding'], dtype=np.float32)
                    training_data.append(data)
                return training_data
        except Exception as e:
            logging.error(f"❌ Ошибка получения данных обучения: {e}")
            return []
    
    def save_model_metrics(self, model_name: str, metrics: Dict[str, Any]) -> bool:
        """Сохраняет метрики модели"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_metrics 
                    (model_name, accuracy, precision_score, recall_score, f1_score, training_examples)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    model_name,
                    metrics.get('accuracy', 0.0),
                    metrics.get('precision', 0.0),
                    metrics.get('recall', 0.0),
                    metrics.get('f1', 0.0),
                    metrics.get('training_examples', 0)
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"❌ Ошибка сохранения метрик: {e}")
            return False
    
    def get_latest_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Получает последние метрики модели"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM model_metrics 
                    WHERE model_name = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ''', (model_name,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logging.error(f"❌ Ошибка получения метрик: {e}")
            return None
    
    def update_daily_stats(self, date: str, stats: Dict[str, int]) -> bool:
        """Обновляет дневную статистику"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO bot_stats 
                    (date, messages_processed, messages_forwarded, messages_rejected, training_examples_added)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    date,
                    stats.get('processed', 0),
                    stats.get('forwarded', 0),
                    stats.get('rejected', 0),
                    stats.get('training_examples', 0)
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"❌ Ошибка обновления статистики: {e}")
            return False
    
    def get_stats_summary(self, days: int = 7) -> Dict[str, Any]:
        """Получает сводную статистику за последние дни"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        SUM(messages_processed) as total_processed,
                        SUM(messages_forwarded) as total_forwarded,
                        SUM(messages_rejected) as total_rejected,
                        SUM(training_examples_added) as total_training,
                        COUNT(*) as active_days
                    FROM bot_stats 
                    WHERE date >= date('now', '-{} days')
                '''.format(days))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'total_processed': row['total_processed'] or 0,
                        'total_forwarded': row['total_forwarded'] or 0,
                        'total_rejected': row['total_rejected'] or 0,
                        'total_training': row['total_training'] or 0,
                        'active_days': row['active_days'] or 0,
                        'forward_rate': (row['total_forwarded'] or 0) / max(row['total_processed'] or 1, 1)
                    }
                return {}
        except Exception as e:
            logging.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def clear_old_data(self, days: int = 30) -> bool:
        """Очищает старые данные"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM messages 
                    WHERE created_at < datetime('now', '-{} days')
                '''.format(days))
                deleted_messages = cursor.rowcount
                
                cursor.execute('''
                    DELETE FROM bot_stats 
                    WHERE date < date('now', '-{} days')
                '''.format(days))
                deleted_stats = cursor.rowcount
                
                conn.commit()
                logging.info(f"✅ Очищено {deleted_messages} сообщений и {deleted_stats} записей статистики")
                return True
        except Exception as e:
            logging.error(f"❌ Ошибка очистки данных: {e}")
            return False
