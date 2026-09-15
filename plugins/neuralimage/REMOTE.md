# Удалённая работа NeuralImage

## Установка

Из корня репозитория, в отдельном окружении для каждой поставки:

```sh
# Полная программа: локальное и удалённое выполнение
uv sync --frozen --package neuralimage --extra full --no-dev
# Лёгкий Qt-клиент
uv sync --frozen --package neuralimage --extra client --no-dev
# Вычислительный сервер
uv sync --frozen --package neuralimage --extra server --no-dev
```

Запуск: `neuralimage`, `neuralimage-client`, `neuralimage-server` из каталога
`.venv/Scripts` (Windows) или `.venv/bin` (Linux). `uv sync` синхронизирует выбранное
окружение: для всего Kraken используйте отдельную синхронизацию всех пакетов и extras.

Клиент сохраняет Qt, NumPy, Pillow и библиотеки обработки изображений для
редакторов и просмотра. PyTorch, torchvision, timm и CUDA в него не входят.
Сервер использует headless OpenCV и не устанавливает Qt.

## Docker с NVIDIA GPU

Нужны Docker с передачей NVIDIA GPU, совместимый драйвер на хосте и NVIDIA
Container Toolkit на Linux. На Windows используйте Docker Desktop с работающей
передачей GPU в Linux-контейнеры.

```sh
docker compose -f plugins/neuralimage/docker/compose.yaml up --build -d
docker compose -f plugins/neuralimage/docker/compose.yaml logs -f
```

Порт `8765`, постоянный volume `neuralimage-data`. Сервер проверяет CUDA при
запуске и завершается с понятной ошибкой, если GPU недоступен. Один сервер
выполняет задания последовательно; несколько HTTP workers не поддерживаются.
Блокировка каталога исключает запуск двух серверов над одной очередью.

Согласованный режим — локальная сеть без авторизации. Все устройства с доступом
к порту могут просматривать задания и управлять ими. Адрес клиента:
`http://ИМЯ-ИЛИ-IP-СЕРВЕРА:8765`.

Запуск без Docker и просмотр заданий:

```sh
neuralimage-server --data-dir ./neuralimage-data serve --host 0.0.0.0 --port 8765
neuralimage-server --data-dir ./neuralimage-data jobs
```

`--max-job-gib` ограничивает суммарный размер входов задания (100 GiB по умолчанию).
`--retention-days 0` сохраняет данные до явной очистки. Положительное значение
включает очистку старых завершённых заданий при запуске сервера.

Для явной очистки остановите сервер и выполните:

```sh
neuralimage-server --data-dir ./neuralimage-data cleanup --older-than-days 30
```

Удаляются успешные, ошибочные и отменённые задания старше указанного срока.
Приостановленные и прерванные задания сохраняются.

## Работа клиента

1. Выберите «На сервере», укажите URL и проверьте соединение. В лёгкой сборке
   удалённый режим включён постоянно.
2. Выберите обычные локальные папки, модель и параметры. Папка результата
   остаётся локальной. После отправки параметры задания зафиксированы.
3. Клиент загружает файлы и показывает события сервера: прогресс, графики,
   промежуточные изображения, ошибки и запросы подтверждения.
4. После завершения результаты скачиваются в подкаталог с ID задания внутри
   выбранной папки результата. SHA-256 проверяется до замены конечных файлов.

Передаются исходные кадры; нарезка выполняется на сервере. Checkpoint рядом
с моделью и вспомогательные файлы внутри выборки также передаются. Неизвестные
внешние файловые ссылки во вложенных настройках отклоняются с ошибкой.
Предпросмотр вычисляется общим CPU-модулем сервера по нужным образцам, включая
партнёра MixUp. Устаревший ответ не заменяет изображение для новых настроек.

Закрытие клиента не отменяет задачу. В папке результата сохраняется
`.neuralimage-remote-*.json`. Кнопка «Восстановить удалённую задачу» открывает этот
файл, возобновляет загрузку или наблюдение и повторяет скачивание. Восстановление
приостановленного задания продолжает его выполнение. После аварии сервера
продолжение доступно при наличии checkpoint или манифеста распознавания.

Ошибка скачивания не меняет успешный статус вычислений. Повторить скачивание
можно также через CLI:

```sh
neuralimage-server download --url http://SERVER:8765 --job JOB_ID --destination ./results
```

## Сборка клиента Windows

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD/.client-build-env"
uv sync --frozen --package neuralimage --extra client --extra build --no-dev
.client-build-env/Scripts/python.exe -m PyInstaller --noconfirm --distpath plugins/neuralimage/dist plugins/neuralimage/packaging/NeuralImageClient.spec
ISCC plugins/neuralimage/packaging/NeuralImageClientInstaller.iss
```

Установщик имеет отдельный AppId и регистрирует обработчик `neuralimage` в
Kraken. Полная и лёгкая сборки используют один идентификатор плагина: последняя
установленная версия становится зарегистрированным обработчиком.

## Протокол и проверки

API `/api/v1`: `capabilities`, `jobs`, `preview`; операции задания:
`inputs/{path}`, `submit`, `events?after=SEQ`, `pause`, `resume`, `cancel`,
`answer`, `artifacts`, `artifacts/{path}`. Создание идемпотентно по `request_key`.
PUT передаёт части до 4 MiB; GET возвращает offset. После `submit` входы закрыты.
Статус `finishing` означает ожидание фактического выхода вычислительного процесса.

Тесты протокола: `test_remote_execution.py`. Предпросмотр и защита от устаревших
ответов: `test_remote_preview.py`. Реальный GPU-тест `test_remote_gpu.py`
запускается с `NEURALIMAGE_RUN_GPU_TESTS=1`: обучение, распознавание через HTTP
и сравнение растров с локальным запуском. Docker проверяется отдельно на хосте
с настроенной передачей GPU.
