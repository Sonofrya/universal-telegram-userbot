"""
Модуль машинного обучения для классификации сообщений
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from database import DatabaseManager
from config import config

class UniversalMessageClassifier:
    """Универсальный классификатор сообщений с автообучением"""
    
    def __init__(self, model_name: str = None, db_manager: DatabaseManager = None):
        self.model_name = model_name or config.ml.classifier_model
        self.db_manager = db_manager or DatabaseManager()
        self.classifier = None
        self.sentence_model = None
        self.is_trained = False
        self.training_data = []
        self.last_metrics = {}
        
        # Загружаем модель предложений
        self._load_sentence_model()
        # Загружаем данные обучения
        self._load_training_data()
    
    def _load_sentence_model(self):
        """Загружает модель для создания эмбеддингов"""
        try:
            self.sentence_model = SentenceTransformer(config.ml.model_name)
            logging.info(f"✅ Модель предложений загружена: {config.ml.model_name}")
        except Exception as e:
            logging.error(f"❌ Ошибка загрузки модели предложений: {e}")
            self.sentence_model = None
    
    def _load_training_data(self):
        """Загружает данные обучения из базы данных"""
        try:
            self.training_data = self.db_manager.get_training_data()
            logging.info(f"✅ Загружено {len(self.training_data)} примеров для обучения")
            
            # Загружаем последние метрики
            self.last_metrics = self.db_manager.get_latest_metrics(self.model_name) or {}
            
        except Exception as e:
            logging.error(f"❌ Ошибка загрузки данных обучения: {e}")
            self.training_data = []
    
    def add_training_example(self, text: str, label: int) -> bool:
        """Добавляет пример для обучения"""
        if not self.sentence_model:
            logging.error("❌ Модель предложений не загружена")
            return False
        
        try:
            # Создаем эмбеддинг
            embedding = self.sentence_model.encode([text])[0]
            
            # Сохраняем в базу данных
            success = self.db_manager.save_training_example(text, embedding, label)
            
            if success:
                # Обновляем локальные данные
                self.training_data.append({
                    'text': text,
                    'embedding': embedding,
                    'label': label
                })
                
                logging.info(f"✅ Добавлен пример обучения (всего: {len(self.training_data)})")
                
                # Автоматическое обучение при накоплении примеров
                if len(self.training_data) >= config.ml.min_training_examples:
                    if len(self.training_data) % config.ml.auto_train_threshold == 0:
                        self.auto_train()
                
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"❌ Ошибка добавления примера обучения: {e}")
            return False
    
    def auto_train(self) -> bool:
        """Автоматическое обучение модели"""
        if len(self.training_data) < config.ml.min_training_examples:
            logging.warning(f"❌ Недостаточно данных для обучения (нужно {config.ml.min_training_examples}, есть {len(self.training_data)})")
            return False
        
        try:
            # Подготавливаем данные
            X = np.array([item['embedding'] for item in self.training_data])
            y = np.array([item['label'] for item in self.training_data])
            
            # Проверяем баланс классов
            unique_labels, counts = np.unique(y, return_counts=True)
            if len(unique_labels) < 2:
                logging.warning("❌ Недостаточно классов для обучения")
                return False
            
            # Создаем новую модель если нужно
            if self.classifier is None:
                self.classifier = LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    class_weight='balanced'  # Для балансировки классов
                )
            
            # Обучаем модель
            self.classifier.fit(X, y)
            self.is_trained = True
            
            # Рассчитываем метрики
            metrics = self._calculate_metrics(X, y)
            
            # Сохраняем метрики в базу данных
            self.db_manager.save_model_metrics(self.model_name, metrics)
            self.last_metrics = metrics
            
            logging.info("✅ Модель автоматически переобучена!")
            logging.info(f"📊 Метрики: Точность: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Ошибка автоматического обучения: {e}")
            return False
    
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Рассчитывает метрики модели"""
        try:
            y_pred = self.classifier.predict(X)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
                'training_examples': len(self.training_data)
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"❌ Ошибка расчета метрик: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'training_examples': 0}
    
    def predict(self, text: str) -> Optional[float]:
        """Предсказывает вероятность для текста"""
        if not self.is_trained or not self.classifier or not self.sentence_model:
            return None
        
        try:
            # Создаем эмбеддинг
            embedding = self.sentence_model.encode([text])[0]
            
            # Предсказываем вероятность
            probability = self.classifier.predict_proba([embedding])[0][1]
            return float(probability)
            
        except Exception as e:
            logging.error(f"❌ Ошибка предсказания: {e}")
            return None
    
    def predict_batch(self, texts: List[str]) -> List[Optional[float]]:
        """Предсказывает вероятности для списка текстов"""
        if not self.is_trained or not self.classifier or not self.sentence_model:
            return [None] * len(texts)
        
        try:
            # Создаем эмбеддинги
            embeddings = self.sentence_model.encode(texts)
            
            # Предсказываем вероятности
            probabilities = self.classifier.predict_proba(embeddings)[:, 1]
            return [float(p) for p in probabilities]
            
        except Exception as e:
            logging.error(f"❌ Ошибка batch предсказания: {e}")
            return [None] * len(texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получает статистику модели"""
        stats = {
            'is_trained': self.is_trained,
            'training_examples': len(self.training_data),
            'model_name': self.model_name,
            'sentence_model_loaded': self.sentence_model is not None
        }
        
        # Добавляем метрики если они есть
        if self.last_metrics:
            stats.update({
                'accuracy': self.last_metrics.get('accuracy', 0.0),
                'precision': self.last_metrics.get('precision_score', 0.0),
                'recall': self.last_metrics.get('recall_score', 0.0),
                'f1_score': self.last_metrics.get('f1_score', 0.0),
                'last_training_date': self.last_metrics.get('created_at', '')
            })
        
        return stats
    
    def get_training_data_stats(self) -> Dict[str, Any]:
        """Получает статистику данных обучения"""
        if not self.training_data:
            return {'total': 0, 'positive': 0, 'negative': 0, 'balance': 0.0}
        
        labels = [item['label'] for item in self.training_data]
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        
        return {
            'total': len(self.training_data),
            'positive': positive_count,
            'negative': negative_count,
            'balance': positive_count / len(self.training_data) if self.training_data else 0.0
        }
    
    def retrain(self) -> bool:
        """Принудительное переобучение модели"""
        logging.info("🔄 Начинаем принудительное переобучение...")
        return self.auto_train()
    
    def export_model(self, filepath: str) -> bool:
        """Экспортирует модель в файл"""
        try:
            import pickle
            
            model_data = {
                'classifier': self.classifier,
                'model_name': self.model_name,
                'training_data_count': len(self.training_data),
                'metrics': self.last_metrics,
                'exported_at': str(np.datetime64('now'))
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"✅ Модель экспортирована в {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Ошибка экспорта модели: {e}")
            return False
    
    def import_model(self, filepath: str) -> bool:
        """Импортирует модель из файла"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.model_name = model_data.get('model_name', self.model_name)
            self.is_trained = True
            self.last_metrics = model_data.get('metrics', {})
            
            logging.info(f"✅ Модель импортирована из {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Ошибка импорта модели: {e}")
            return False
