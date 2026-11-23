# Постоянный beta-стенд (Server + PostgreSQL + Blob Gateway + agent)

Воспроизводимый стенд для слоёв проверки `integration`, `gui`/`stateful` (server-backed)
и `live` из `scripts/quality-gates.ps1`. Скрипты: `scripts/start-beta-stand.ps1` и
`scripts/stop-beta-stand.ps1`; вспомогательная логика секретов/БД/HTTP —
`tools/quality/beta_stand.py` (PowerShell 5.1 не умеет DPAPI/TOML, см. заметку о
нативных командах в `scripts/quality-gates.ps1`).

## Главное правило

**Рабочую `kraken_local` не трогать.** Стенд создаёт ТОЛЬКО выделенную БД
`kraken_beta_test`, никогда не пишет в схему `public` рабочей базы и не меняет
`config/local/server.toml` — он читается как шаблон, а рабочая копия стенда лежит в
`.test_runtime\beta-stand\server.toml` (git-ignored). Секреты стенда (DPAPI-файлы,
токены, env json) тоже только под `.test_runtime\beta-stand*`.

## Bring-up

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-beta-stand.ps1
```

Идемпотентен: повторный запуск сходится (живой Server/агент переиспользуются, БД не
сбрасывается, пока Server держит её; сброс `--reset` выполняется только когда Server
не запущен). `-Restart` = полный teardown через `stop-beta-stand.ps1` и чистый подъём.
По каждому компоненту печатается `PASS`/`FAIL: <причина>`; env-выходы пишутся даже
при отсутствующем Blob Gateway (интеграционному слою нужен только URL БД).

| Компонент | Что делает старт | Остановка |
|---|---|---|
| PostgreSQL | `kraken-admin`-совместимая БД `kraken_beta_test` (см. «Fallback: схема» ниже), DPAPI-секрет `beta-database-url.secret` | `DROP DATABASE` (или `DROP SCHEMA ... CASCADE`) |
| Server | `.venv\Scripts\python.exe -m kraken_server --config .test_runtime\beta-stand\server.toml`, health `http://127.0.0.1:8080/api/v1/health` | PID из `state.json` (проверка по command line), порт 8080 закрывается |
| Blob Gateway | запускает сам Server (`[blob_gateway] enabled=true`, Rust exe — см. ниже); health `http://127.0.0.1:8081/health` | дочерний процесс Server'а; PID в `state.json`, порт 8081 |
| Аккаунт/сессия | регистрация/вход loopback `POST /api/v1/auth/accounts` → `session.secret` (DPAPI) | удаляется вместе с БД |
| Protected-проект | `beta-stand-protected` (2×1) — сентинел неизменности в fixture | удаляется вместе с БД |
| Machine token | `kraken-admin create-agent-token --capability frames.vectorize.v1` → `machine.secret` (DPAPI), id в `state.json` | `kraken-admin revoke-agent-token` + **proof: lease → HTTP 401** (пока Server ещё жив) |
| Dataset token | `create-agent-token --capability frames.dataset.prepare.v1` → `dataset.secret` (DPAPI), id `dataset_machine_token_id` | `revoke-agent-token` при остановке стенда |
| NeuralImage token | `create-agent-token --capability frames.binary-segment.v2` → `neuralimage.secret` (DPAPI), id `neuralimage_machine_token_id` | `revoke-agent-token` при остановке стенда |
| Agent (опция `-WithWorkerAgent`) | `python -m kraken_agent --server-url ... --plugins-config worker-plugins.json --independent`, токен через env `KRAKEN_AGENT_TOKEN` | PID из `state.json` |

Стенд переживает закрытие консоли, из которой запущен: компоненты стартуют
двухступенчатым `cmd /c start` (orphans), иначе `kraken_core.process_lifetime`
завершил бы Server вместе с деревом запускающего шелла. Останавливать стенд —
только `stop-beta-stand.ps1`.

**Почему agent по умолчанию выключен.** `tests/test_live_server_worker_acceptance.py`
сам исполняет контролируемый one-shot machine worker; параллельный agent c глобальным
polling lease возьмёт его задание («Unexpected lease; stop acceptance»). Фоновый
worker (`-WithWorkerAgent`) включают только задачи автономного compute-плана, на
время live-приёмки его останавливают.

### Выходы для вызывающего

- `.test_runtime\beta-stand\live-worker-config.json` — контракт
  `KRAKEN_LIVE_WORKER_CONFIG` (`server_url`, `session_secret`, `machine_secret`,
  `dataset_machine_secret`, `neuralimage_machine_secret`, `model_reuse_receipt`,
  `protected_project_id`, `results_directory`; значения секретов в JSON запрещены,
  только DPAPI-пути).
- `.test_runtime\beta-stand.env.json` — `KRAKEN_TEST_POSTGRES_URL` (plaintext URL —
  git-ignored location) и путь `KRAKEN_LIVE_WORKER_CONFIG`.

```powershell
$repo = (Resolve-Path .).Path
$env:KRAKEN_LIVE_WORKER_CONFIG = Join-Path $repo '.test_runtime\beta-stand\live-worker-config.json'
$env:KRAKEN_TEST_POSTGRES_URL = (Get-Content (Join-Path $repo '.test_runtime\beta-stand.env.json') -Raw | ConvertFrom-Json).KRAKEN_TEST_POSTGRES_URL
$env:KRAKEN_MODEL_REUSE_RECEIPT = Join-Path $repo '.test_runtime\beta-stand\results\model-reuse-receipt.json'
```

## KRAKEN_UI_TEST_CONFIG для палитры W3

Контракт `tests/test_live_ui_workflow.py` (header docstring): JSON-файл с ключами
`server_url`, `username`, `password`, `source_directory`; в `source_directory`
ровно **100 пронумерованных изображений** (`000.png`..`099.png`, grayscale 64×64).
Файлы-источник и конфиг генерируются throwaway-скриптом (git-ignored, не коммитится):

```powershell
.venv\Scripts\python.exe .test_runtime\make-ui-test-config.py
# -> .test_runtime\palette-source-zero-based\000.png..099.png (100 файлов)
# -> .test_runtime\beta-stand\ui-test-config.json (server_url/username/password
#    из state.json стенда; source_directory — прямые слеши, POSIX-стиль)
```

Скрипт перечитывает `.test_runtime\beta-stand\state.json` (`utf-8-sig` — PowerShell
пишет с BOM), поэтому после `-Restart`/`--reset` пересоздания аккаунта достаточно
запустить его ещё раз — конфиг сойдётся с новыми учётными данными.

Канонический преамбула-вызов для сценариев палитры (используется задачами W3
дословно, подставляя дату доказательства и команду):

```powershell
$repo = (Resolve-Path .).Path
$env:KRAKEN_LIVE_WORKER_CONFIG = Join-Path $repo '.test_runtime\beta-stand\live-worker-config.json'
$env:KRAKEN_TEST_POSTGRES_URL = (Get-Content (Join-Path $repo '.test_runtime\beta-stand.env.json') -Raw | ConvertFrom-Json).KRAKEN_TEST_POSTGRES_URL
$env:KRAKEN_UI_TEST_CONFIG = Join-Path $repo '.test_runtime\beta-stand\ui-test-config.json'
$env:QT_QPA_PLATFORM = 'offscreen'
# Запустите pytest или quality-gates.ps1 в этой же PowerShell-сессии.
```

**Учётные записи.** Стенд провизнит ОДНУ учётную запись (`beta-stand`,
`state.json`). Дополнительные пользователи сценарии регистрируют сами через
`POST /api/v1/auth/accounts` на loopback — регистрация открыта при
`domain_auth=false` (см. `src/kraken_server/interface/http/routes/auth.py`);
роль в проекте выдаётся затем ACL-механизмом самого сценария.

## Teardown

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop-beta-stand.ps1   # -KeepDatabase — не удалять БД
```

Порядок: revoke machine token (пока Server отвечает) → proof 401 на
`POST /api/v1/agent/lease` → остановка agent/server/gateway (PID сверяются по
command line, переиспользованные PID чужих процессов не убиваются) → ожидание
закрытия 8080/8081 → `DROP` тестовой БД/схемы → удаление секретов/env/state
(логи и результаты остаются в `.test_runtime\beta-stand`).

## What env each gate tier needs

| Tier (`quality-gates.ps1`) | Нужно со стенда | Дополнительно |
|---|---|---|
| `unit`, `smoke`, `regression`, `registry` | ничего | работают без стенда |
| `integration` | `KRAKEN_TEST_POSTGRES_URL` | PG-гейт-файлы сами скипаются без переменной |
| `gui` | ничего (offscreen Qt) | server-backed варианты живут в live-слое |
| `stateful` | `KRAKEN_TEST_POSTGRES_URL` (когда появится, Task 3.x) | — |
| `live` | `KRAKEN_LIVE_WORKER_CONFIG` | файлы с собственными гейтами: `test_live_server_plugin_chain.py`, `test_live_server_worker_acceptance.py` (стенд); `test_live_ui_workflow.py` — `KRAKEN_UI_TEST_CONFIG`; `test_pipeline_live_contour.py` — `KRAKEN_RUN_PIPELINE_LIVE`; `test_pipeline_live_neural_model.py` — `KRAKEN_RUN_MODEL_FUNCTIONAL`; `test_neuralimage_headless_training.py` — `KRAKEN_RUN_TRAINING_LIVE` |

Gate сам стенд не поднимает: переменные наследуются из окружения вызывающего.

## Convention: два прогона подряд

Приёмка стенда — **один и тот же прогон дважды подряд без перезагрузки стенда**
(конвенция плана стабилизации): результаты обязаны совпасть, идемпотентность
заданий/lease проверяется именно повтором.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_live_server_worker_acceptance.py -q   # прогон 1
.venv\Scripts\python.exe -m pytest tests/test_live_server_worker_acceptance.py -q   # прогон 2
```

Ожидаемо: exit 0, без skip, дважды. Skips считаются PARTIAL, не PASS.

## Blob Gateway: состояние на закрытых сетях

Acceptance требует `transfer["mode"] == "gateway"` — обязателен Rust exe
`blob_gateway\target\release\kraken-blob-gateway.exe`. **С 19.11.2026 exe собран и
поставки с носителя не требуется:** вендор крейтов отзеркален в
`C:\code\kraken\.offline\cargo-vendor`, `.cargo\config.toml` настраивает
`replace-with = 'kraken-offline-vendor'`, и сборка полностью офлайн:

```powershell
cargo build --release --offline --manifest-path blob_gateway\Cargo.toml
```

Проверено: sha256 `blob_gateway\Cargo.lock` совпадает с `Cargo.lock.sha256` комплекта;
сборка `Finished release profile` exit 0; стенд со шлюзом `PASS gateway`;
`tests/test_live_server_worker_acceptance.py` — `1 passed`, exit 0, дважды подряд
(gateway-режим). Каноническая поставка вендора для свежей машины —
`offline/Create-OfflineKit.ps1` на машине с сетью → `dependencies/cargo/vendor`.

Если exe отсутствует, стенд остаётся в proxy-режиме: `integration`-слой работает
полностью, live-приёмка падает на `assert transfer["mode"] == "gateway"` — это блокер
среды, а не продукта.

## Fallback: схема вместо базы (зафиксированная замена)

Роль приложения в `config/local/database-url.secret` — `kraken_local_app`,
`rolsuper=false`, `rolcreatedb=false`: `CREATE DATABASE kraken_beta_test` даёт
«нет прав на создание базы данных». Скрипт автоматически падает в документированный
fallback: **SCHEMA `kraken_beta_test` внутри базы `kraken_local`** с
`options=-csearch_path=kraken_beta_test`; данные рабочей схемы `public` не читаются
и не пишутся, `DROP SCHEMA ... CASCADE` на teardown удаляет только объекты стенда.
Если появится роль с `CREATEDB` (или superuser-секрет), тот же скрипт вернётся к
отдельной базе без изменений — режим пишется в `state.json` (`db_mode`).

## S14: резервное копирование сервера

`tests/test_server_backup.py` создаёт и удаляет две базы `kraken_backup_test_*`.
Для этого нужны `KRAKEN_TEST_PG_BIN` (`pg_dump`/`pg_restore` рядом с кластером) и
роль с `CREATEDB`. Роль стенда `kraken_local_app` такого права не имеет, секрет
суперпользователя на закрытой машине не запрашивался. Сценарий
В постоянном стенде тесты S14 пропускаются из-за отсутствия `CREATEDB`.
Пароль роли не выдумывается и в репозиторий не записывается.

Это ограничение относится к постоянному beta-стенду. Для S14 уже есть
[изолированный сценарий](../operations/runbook.md#изолированный-тест-backuprestore-s14),
который поднимает временный кластер с ролью `kraken_audit`, задаёт обе тестовые
переменные и останавливает кластер после прогона. 20.11.2026 сценарий прошёл
дважды: по 26 PASS, без пропусков. JUnit, pytest и receipt сохранены в
`evidence/2026-11-20/s14/run-{1,2}/`. Эта проверка подтверждает S14 на
изолированном кластере; основной integration tier остаётся PARTIAL.

## Замечание D21 (перенесено обязательно)

`UV_CACHE_DIR=G:\kraken-transfer\uv-cache` accelerates offline uv resolution on
closed-network machines (contour build deps; cu132 index pages not cached).

С 19.11.2026 кэш отзеркален локально в проект — подключать комплект для офлайн-резолва
не нужно: `UV_CACHE_DIR=C:\code\kraken\.offline\uv-cache` (git-игнорируемая папка;
там же `native-wheels\contour-0.9.10-cp314-win_amd64.whl`, `tools\uv.exe`,
`tools\install-buildtools.ps1`, `kit-docs\`). Полная пересборка `uv.lock` по-прежнему
требует один сетевой доступ к download.pytorch.org (страницы индекса cu132 не кэшируются).

## Свойства идемпотентности (проверено)

- **Guard имени БД/схемы:** `-DatabaseName` проверяется ДО любого DDL
  (`check-database-name`, та же функция в `db-provision`/`db-drop`): отказать
  рабочему `kraken_local` (из base URL), `postgres`, `template*`, а в режиме
  схемы — `public`, `information_schema`, `pg_*`. `FAIL preconditions`,
  exit 1, ни одного подключения. Покрыто `tests/test_beta_stand_guard.py`.

- Старт при живом стенде: все компоненты `PASS`/reused, БД не сбрасывается.
- Старт после падения Server'а: БД сбрасывается (`--reset`), миграции до head,
  аккаунт/проект/токены пересоздаются.
- Stop без state.json: no-op exit 0. Stop при мёртвом Server: revoke выполняется по
  БД, proof 401 помечается пропущенным.
- Повторный stop: токен «already revoked», БД «dropped» — сходится.
