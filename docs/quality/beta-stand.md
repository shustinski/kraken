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
  `protected_project_id`, `results_directory`; значения секретов в JSON запрещены,
  только DPAPI-пути).
- `.test_runtime\beta-stand.env.json` — `KRAKEN_TEST_POSTGRES_URL` (plaintext URL —
  git-ignored location) и путь `KRAKEN_LIVE_WORKER_CONFIG`.

```powershell
$env:KRAKEN_LIVE_WORKER_CONFIG  = 'C:\code\kraken\.test_runtime\beta-stand\live-worker-config.json'
$env:KRAKEN_TEST_POSTGRES_URL  = (Get-Content .test_runtime\beta-stand.env.json -Raw | ConvertFrom-Json).KRAKEN_TEST_POSTGRES_URL
```

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
`blob_gateway\target\release\kraken-blob-gateway.exe`. На закрытой машине без
сетевых источников его построить нельзя: crates.io недоступен, локального кэша
реестра нет. Каноническая поставка — offline-kit: `offline/Create-OfflineKit.ps1`
на машине с сетью генерирует `dependencies/cargo/vendor` (из `Cargo.lock`) и
`config.toml` с `replace-with = 'offline-vendor'`; после переноса комплекта сборка
на закрытой машине:

```powershell
cargo build --release --offline --manifest-path blob_gateway\Cargo.toml
```

Либо перенести готовый `kraken-blob-gateway.exe` и положить по пути из
`server.toml`/в `blob_gateway\target\release\`, затем перезапустить стенд. Без exe
стенд остаётся в proxy-режиме: `integration`-слой работает полностью, live-приёмка
падает на `assert transfer["mode"] == "gateway"` — это блокер среды, а не продукта.

## Fallback: схема вместо базы (зафиксированная замена)

Роль приложения в `config/local/database-url.secret` — `kraken_local_app`,
`rolsuper=false`, `rolcreatedb=false`: `CREATE DATABASE kraken_beta_test` даёт
«нет прав на создание базы данных». Скрипт автоматически падает в документированный
fallback: **SCHEMA `kraken_beta_test` внутри базы `kraken_local`** с
`options=-csearch_path=kraken_beta_test`; данные рабочей схемы `public` не читаются
и не пишутся, `DROP SCHEMA ... CASCADE` на teardown удаляет только объекты стенда.
Если появится роль с `CREATEDB` (или superuser-секрет), тот же скрипт вернётся к
отдельной базе без изменений — режим пишется в `state.json` (`db_mode`).

## Замечание D21 (перенесено обязательно)

`UV_CACHE_DIR=G:\kraken-transfer\uv-cache` accelerates offline uv resolution on
closed-network machines (contour build deps; cu132 index pages not cached).

## Свойства идемпотентности (проверено)

- Старт при живом стенде: все компоненты `PASS`/reused, БД не сбрасывается.
- Старт после падения Server'а: БД сбрасывается (`--reset`), миграции до head,
  аккаунт/проект/токены пересоздаются.
- Stop без state.json: no-op exit 0. Stop при мёртвом Server: revoke выполняется по
  БД, proof 401 помечается пропущенным.
- Повторный stop: токен «already revoked», БД «dropped» — сходится.
