# 🚀 Руководство по развертыванию

## 📋 Требования

- Python 3.8+
- Telegram API ключи
- SQLite (включен в Python)

## 🖥️ Локальное развертывание

### 1. Подготовка окружения
```bash
# Клонирование репозитория
git clone https://github.com/yourusername/universal-telegram-bot.git
cd universal-telegram-bot

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка
```bash
# Автоматическая настройка
python setup.py

# Или ручная настройка
cp .env.example .env
# Отредактируйте .env файл
```

### 3. Запуск
```bash
python main_universal.py
```

## ☁️ Развертывание на сервере

### Docker (рекомендуется)

#### 1. Создайте Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main_universal.py"]
```

#### 2. Создайте docker-compose.yml
```yaml
version: '3.8'

services:
  telegram-bot:
    build: .
    container_name: universal-telegram-bot
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1
```

#### 3. Запуск
```bash
docker-compose up -d
```

### Systemd (Linux)

#### 1. Создайте сервис
```bash
sudo nano /etc/systemd/system/telegram-bot.service
```

#### 2. Конфигурация сервиса
```ini
[Unit]
Description=Universal Telegram Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/universal-telegram-bot
ExecStart=/path/to/venv/bin/python main_universal.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 3. Запуск сервиса
```bash
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot
sudo systemctl status telegram-bot
```

## 🔧 Настройка продакшена

### 1. Безопасность
```bash
# Установите правильные права доступа
chmod 600 .env
chmod 600 session.txt

# Создайте отдельного пользователя
sudo useradd -m -s /bin/bash telegram-bot
sudo chown -R telegram-bot:telegram-bot /path/to/bot
```

### 2. Логирование
```bash
# Настройте ротацию логов
sudo nano /etc/logrotate.d/telegram-bot
```

```conf
/path/to/universal-telegram-bot/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 telegram-bot telegram-bot
}
```

### 3. Мониторинг
```bash
# Установите мониторинг
pip install psutil

# Создайте скрипт мониторинга
nano monitor.py
```

```python
import psutil
import time
import subprocess

def check_bot_status():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'main_universal.py' in ' '.join(proc.info['cmdline']):
            return True
    return False

if __name__ == "__main__":
    if not check_bot_status():
        print("Bot is down, restarting...")
        subprocess.run(["systemctl", "restart", "telegram-bot"])
```

## 📊 Мониторинг и логи

### Логи
```bash
# Просмотр логов
tail -f universal_bot.log

# Логи systemd
journalctl -u telegram-bot -f
```

### Метрики
- Используйте команду `/stats` в боте
- Проверяйте файл базы данных
- Мониторьте использование ресурсов

## 🔄 Обновления

### 1. Остановка сервиса
```bash
sudo systemctl stop telegram-bot
```

### 2. Обновление кода
```bash
git pull origin main
pip install -r requirements.txt
```

### 3. Запуск сервиса
```bash
sudo systemctl start telegram-bot
```

## 🛠️ Устранение неполадок

### Проблемы с Telegram API
```bash
# Проверьте настройки
cat .env | grep TELEGRAM

# Пересоздайте сессию
rm session.txt
python main_universal.py
```

### Проблемы с базой данных
```bash
# Проверьте права доступа
ls -la bot_database.db

# Пересоздайте базу данных
rm bot_database.db
python main_universal.py
```

### Проблемы с производительностью
```bash
# Мониторинг ресурсов
htop
df -h
free -h
```

## 📞 Поддержка

При проблемах с развертыванием:
1. Проверьте логи
2. Убедитесь в правильности конфигурации
3. Создайте issue в репозитории
4. Приложите логи и информацию о системе
