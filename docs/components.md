# Компоненты и режимы

| Компонент | Ответственность | Запуск |
|---|---|---|
| Hub / Desktop | Окно, каталог и работа с проектами | `kraken-hub`, установленный Desktop |
| Server | HTTP/WebSocket, проекты, сессии и задания | `kraken-server`, `KrakenServer.exe` или служба |
| PostgreSQL | Постоянные данные общих проектов | Служба PostgreSQL |
| Blob Gateway | Большие загрузки и скачивания | Дочерний Rust-процесс Server при включенном шлюзе |
| Agent | Локальная очередь или серверный worker | `kraken-agent` |
| Плагины | Contour, NeuralImage, Krona, CSliser, Karakal | Отдельные окружения, операции запускает Agent |
| Admin | Первичная настройка, аккаунты, токены, диагностика | `kraken-admin` на машине Server |

| Режим | Данные | Старт |
|---|---|---|
| Локальный Desktop | Файлы проекта и локальная SQLite | `bootstrap-local`, затем Hub |
| Server `--development` | Временное состояние в памяти | Только отладка API; любой непустой Bearer-токен считается администратором |
| Постоянный Server | PostgreSQL и BlobStore | `kraken-admin init` один раз, затем Server и Hub |
| Beta-стенд | Тестовая БД или отдельная схема | `scripts/start-beta-stand.ps1` |
| Изолированный backup-тест | Временный кластер PostgreSQL с `kraken_audit` | `reproduce-stabilization.ps1 -Block backup` |

Подробности: [запуск](operations/runbook.md), [роли](operations/roles.md), [архитектура](architecture/README.md).
