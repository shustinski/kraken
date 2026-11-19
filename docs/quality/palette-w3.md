# Палитра W3 — индекс живых сценариев принятия

Индекс всех сценариев палитры W3 (план
`docs/superpowers/plans/2026-11-18-beta-stabilization-w3-palette.md`, таблица
«Mapping table» в конце плана; зеркалит колонку «W3 сценарий» чекпоинта
`docs/quality/checkpoint-2026-11-17-beta.md`). Каждая строка — идентифицируемый
блок: сценарий, его env-гейт, его действия реестра, его тестовый файл, его путь
доказательства. Таблица инициализирована на Task 5; статусы и фактические даты
заполняются задачами 6–23 по мере прогонов (конвенция «два прогона подряд» —
`docs/quality/beta-stand.md`).

## Индекс сценариев

Env-гейты: каноническая преамбула вызова — `docs/quality/beta-stand.md`
(«KRAKEN_UI_TEST_CONFIG для палитры W3»); доказательные пути — под
`docs/quality/evidence/<run-date>/` (`--basetemp` — `.test_runtime\`).

| Сценарий | Задача | Env-гейт | Действия реестра | Тестовый файл | Путь доказательства |
|---|---|---|---|---|---|
| S01 | 6 | офскрин GUI (`QT_QPA_PLATFORM=offscreen`, без стенда) | HUB-BOOTSTRAP-ACCOUNT, HUB-LOGIN | `tests/test_beta_palette_hub_local.py` | `…/<run-date>/gate-gui/junit.xml` |
| S02 | 7 | офскрин GUI | HUB-PREFERENCES, HUB-UPDATE-SOURCE, MGR-MATRIX-VISUAL, HUB-CONFIGURE-ROOTS | `tests/test_beta_palette_hub_local.py` | `…/<run-date>/gate-gui/junit.xml` |
| S03 | 8 | офскрин GUI | HUB-BIND-WORKSPACE, HUB-EXPORT-BACKUP, HUB-IMPORT-BACKUP | `tests/test_beta_palette_hub_local.py` | `…/<run-date>/gate-gui/junit.xml` |
| S04 | 9 | офскрин GUI | HUB-SCAN-INTEGRITY, HUB-ADMIN-REFRESH | `tests/test_beta_palette_hub_local.py` | `…/<run-date>/gate-gui/junit.xml` |
| S05 | 10 | `KRAKEN_LIVE_WORKER_CONFIG` (live) | SRV-PROJECTS | `tests/test_beta_palette_server_api.py` | `…/<run-date>/palette/S05-S07/junit.xml` |
| S06 | 10 | `KRAKEN_LIVE_WORKER_CONFIG` (live) | SRV-LAYERS | `tests/test_beta_palette_server_api.py` | `…/<run-date>/palette/S05-S07/junit.xml` |
| S07 | 10 | `KRAKEN_LIVE_WORKER_CONFIG` (live) | SRV-REPRESENTATIONS | `tests/test_beta_palette_server_api.py` | `…/<run-date>/palette/S05-S07/junit.xml` |
| S08 | 11 | `KRAKEN_LIVE_WORKER_CONFIG` (live) | SRV-COMMANDS, SRV-WEBSOCKET, SRV-NOTES-CHECKS-PACKAGES, SRV-JOBS-AGENT-PROTOCOL, SRV-PLANS, SRV-HISTORY-STATISTICS | `tests/test_beta_palette_server_api.py` | `…/<run-date>/palette/S08/junit.xml` |
| S09 | 12 | `KRAKEN_LIVE_WORKER_CONFIG` (live, выделенный agent-токен) | AGT-HEALTH | `tests/test_beta_palette_agent_health.py` | `…/<run-date>/palette/S09/junit.xml` |
| S10 | 13 | `KRAKEN_UI_TEST_CONFIG` (live UI, офскрин) | HUB-LOGIN-TABBED, HUB-CONNECT-SAVED-SERVER, HUB-REFRESH-CATALOG, HUB-OPEN-PROJECT, HUB-RENAME-PROJECT, HUB-ARCHIVE-PROJECT, HUB-RESTORE-PROJECT, HUB-DELETE-PROJECT, HUB-PROJECT-PROPERTIES | `tests/test_beta_palette_hub_ui_project_cycle.py` | `…/<run-date>/palette/S10/junit.xml` |
| S11 | 14 | `KRAKEN_UI_TEST_CONFIG` (live UI, офскрин) | HUB-MATRIX-CONTEXT, HUB-REORDER-LAYERS, HUB-REPRESENTATION-LIFECYCLE (+ in-flow: HUB-ADD-LAYER, HUB-LAYER-CONTEXT, HUB-ADD-REPRESENTATION) | `tests/test_beta_palette_hub_ui_layers.py` | `…/<run-date>/palette/S11/junit.xml` |
| S12 | 15 | `KRAKEN_UI_TEST_CONFIG` (live UI, два пользователя) | HUB-MY-WORK, HUB-WORK-PLAN (+ in-flow: HUB-ISSUE-WORK, HUB-REVIEW-RETURN) | `tests/test_beta_palette_hub_ui_work_flow.py` | `…/<run-date>/palette/S12/junit.xml` |
| S13 | 16 | CLI + `KRAKEN_UI_TEST_CONFIG` (live UI) | HUB-PLUGINS, PLUGIN-CONTOUR-HELP, PLUGIN-NEURALIMAGE-VERSION | `tests/test_plugin_cli_smoke.py`, `tests/test_beta_palette_hub_plugins.py` | `…/<run-date>/gate-smoke/junit.xml`, `…/<run-date>/palette/S13/junit.xml` |
| S14 | 17 | `KRAKEN_TEST_POSTGRES_URL` (+ `KRAKEN_TEST_PG_BIN`) | ADMIN-SERVER-BACKUP-RESTORE (или явное исключение, план Task 17 Step 5) | `tests/test_server_backup.py` | `…/<run-date>/palette/S14/junit.xml` |
| GATE | 18 | предпосылка G-группы (без действий) | — — поставка шлюза + расширение стендовых токенов (`{stage}_machine_secret`, `model_reuse_receipt`) | — | — |
| G16 | 19 | live + Blob Gateway (`transfer.mode == "gateway"`) | BLOB-PUT-GET-RESTART (+ gateway-mode подтверждение PERSISTENT-GATEWAY-UPLOAD, SRV-ARTIFACTS) | `tests/test_blob_gateway.py` | `…/<run-date>/palette/G16/junit.xml` |
| G17 | 20 | live + Blob Gateway | AGT-OP-VECTORIZE | `tests/test_beta_palette_vectorize_multi.py` | `…/<run-date>/palette/G17/junit.xml` (+ `…/<run-date>/gate-live/junit.xml`) |
| G18 | 21 | live + Blob Gateway | AGT-OP-DATASET-PREPARE, PLUGIN-CONTOUR-DATASET-INTERACTIVE | `tests/test_beta_palette_agent_ops.py` | `…/<run-date>/palette/G18/junit.xml` (+ `…/<run-date>/gate-live/junit.xml`) |
| G19 | 22 | live + Blob Gateway + `KRAKEN_RUN_MODEL_FUNCTIONAL=1`, `KRAKEN_RUN_TRAINING_LIVE=1` | AGT-OP-TRAIN, AGT-OP-BINARY-SEGMENT | `tests/test_pipeline_live_neural_model.py`, `tests/test_neuralimage_headless_training.py`, `tests/test_live_server_plugin_chain.py` | `…/<run-date>/palette/G19/junit.xml` |
| FINAL | 23 | полный гейт (все тиры) | coverage-регенерация + live-тайр PASS + чекпоинт + отчёт | `scripts/quality-gates.ps1` | `…/<run-date>/gate-*/` |

Счётная проверка (из плана): 45 NOT_TESTED = S01:2 + S02:4 + S03:3 + S04:2 +
S05:1 + S06:1 + S07:1 + S08:6 + S09:1 + S10:9 + S11:3 + S12:2 + S13:3 + S14:1 +
G16:1 + G17:1 + G18:2 + G19:2 = 45. ✔

## Статус Blob Gateway (состояние на 19.11.2026)

**Поставлен, работает.** Эксплицитной блокировки «exe отсутствует» (текст
Plan Task 18 Step 1 про G:\kraken-transfer / комплект на K:) больше нет — exe
собирается полностью офлайн из отзеркаленного вендора
`C:\code\kraken\.offline\cargo-vendor` (`replace-with =
'kraken-offline-vendor'` в `.cargo\config.toml`):

```powershell
cargo build --release --offline --manifest-path blob_gateway\Cargo.toml
```

- Стенд поднят с включённым шлюзом: `start-beta-stand.ps1` → `BETA STAND RESULT:
  PASS`, строка `PASS gateway` (проверено 19.11.2026).
- Health `http://127.0.0.1:8081/health` → `{"service": "kraken-blob-gateway",
  "status": "ok", …}`.
- G-сценарии (G16–G19) гейтятся не поставкой, а живым стендом с шлюзом и
  контрактом Task 18 (`{stage}_machine_secret`, `model_reuse_receipt` в
  `live-worker-config.json`).

## Политика 15 PASS (гибридное правило, из плана)

HUB-CREATE-PROJECT, HUB-PROJECT-ROLES, HUB-ADD-PERFORMER, HUB-EDIT-PERFORMER,
HUB-DELETE-PERFORMER, HUB-REVIEW-RETURN, HUB-ADD-LAYER, HUB-LAYER-CONTEXT,
HUB-ADD-REPRESENTATION, HUB-ISSUE-WORK, HUB-STATISTICS, SRV-HEALTH-SESSION,
SRV-PRINCIPALS, SRV-ARTIFACTS, PERSISTENT-GATEWAY-UPLOAD — доказательства от
2026-11-18 удовлетворяют post-freeze свежести; каждая строка обязана остаться
PASS на финальном full-gate JUnit (Task 23). Строки, затронутые палитрой
(S11/S12/G16), дополнительно получают палитровые привязки. Падение в финальном
гейте → RED→GREEN триаж, не ручная перепривязка. **HUB-CHECK-UPDATES** остаётся
`BLOCKED` → W4 (тестовый источник обновлений — поставка W4).
