import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TelegramConfig:
    api_id: str
    api_hash: str
    phone_number: str
    session_file: str = 'session.txt'

@dataclass
class MLConfig:
    model_name: str = 'sentence_transformer'
    similarity_threshold: float = 0.7
    min_training_examples: int = 3
    auto_train_threshold: int = 2
    classifier_model: str = 'production_classifier'

@dataclass
class FilterConfig:
    min_message_length: int = 5
    blacklist_words: List[str] = None
    forward_patterns: List[str] = None
    
    def __post_init__(self):
        if self.blacklist_words is None:
            self.blacklist_words = ['спам', 'реклама']
        if self.forward_patterns is None:
            self.forward_patterns = [
                r'пересланное сообщение', r'forwarded message', 
                r'было переслано', r'forwarded from'
            ]

@dataclass
class BusinessConfig:
    keywords: List[str]
    target_user_ids: List[str]
    business_domain: str
    full_cycle_phrases: List[str] = None
    
    def __post_init__(self):
        if self.full_cycle_phrases is None:
            self.full_cycle_phrases = [
                'полный цикл', 'под ключ', 'комплексный', 
                'от и до', 'от концепции до', 'от идеи до'
            ]

class Config:
    
    def __init__(self):
        self.telegram = TelegramConfig(
            api_id=os.getenv('TELEGRAM_API_ID', ''),
            api_hash=os.getenv('TELEGRAM_API_HASH', ''),
            phone_number=os.getenv('TELEGRAM_PHONE', ''),
            session_file=os.getenv('TELEGRAM_SESSION_FILE', 'session.txt')
        )
        
        self.ml = MLConfig(
            model_name=os.getenv('ML_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2'),
            similarity_threshold=float(os.getenv('ML_SIMILARITY_THRESHOLD', '0.7')),
            min_training_examples=int(os.getenv('ML_MIN_TRAINING_EXAMPLES', '3')),
            auto_train_threshold=int(os.getenv('ML_AUTO_TRAIN_THRESHOLD', '2')),
            classifier_model=os.getenv('ML_CLASSIFIER_MODEL', 'production_classifier')
        )
        
        self.filter = FilterConfig(
            min_message_length=int(os.getenv('FILTER_MIN_LENGTH', '5')),
            blacklist_words=self._parse_list(os.getenv('FILTER_BLACKLIST', 'спам,реклама')),
            forward_patterns=self._parse_list(os.getenv('FILTER_FORWARD_PATTERNS', 
                'пересланное сообщение,forwarded message,было переслано'))
        )
        
        self.business = BusinessConfig(
            keywords=self._parse_list(os.getenv('BUSINESS_KEYWORDS', '')),
            target_user_ids=self._parse_list(os.getenv('TARGET_USER_IDS', '')),
            business_domain=os.getenv('BUSINESS_DOMAIN', 'general'),
            full_cycle_phrases=self._parse_list(os.getenv('FULL_CYCLE_PHRASES', 
                'полный цикл,под ключ,комплексный,от и до'))
        )
    
    def _parse_list(self, value: str) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def validate(self) -> bool:
        errors = []
        
        if not self.telegram.api_id:
            errors.append("TELEGRAM_API_ID не установлен")
        if not self.telegram.api_hash:
            errors.append("TELEGRAM_API_HASH не установлен")
        if not self.telegram.phone_number:
            errors.append("TELEGRAM_PHONE не установлен")
        if not self.business.keywords:
            errors.append("BUSINESS_KEYWORDS не установлены")
        if not self.business.target_user_ids:
            errors.append("TARGET_USER_IDS не установлены")
        
        if errors:
            print("❌ Ошибки конфигурации:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_business_config_template(self) -> Dict[str, Any]:
        return {
            'business_domain': 'your_domain',
            'keywords': 'ключевое слово 1, ключевое слово 2, ключевое слово 3',
            'target_user_ids': 'user_id_1,user_id_2',
            'full_cycle_phrases': 'полный цикл,под ключ,комплексный',
            'blacklist_words': 'спам,реклама,нежелательное слово'
        }

config = Config()
