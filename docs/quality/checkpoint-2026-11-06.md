# Продолжение 2026-11-06: checkpoint

План старта: HEAD `4992aa7` (коммит плана), WIP-чекпоинт — `2eeb877`, ветка `main`,
checkout `C:/code/kraken`. План:
[2026-11-06 continuation](../superpowers/plans/2026-11-06-kraken-verification-continuation.md).
Это продолжение стабилизации 2026-11-01: [предыдущий checkpoint](checkpoint-2026-11-01.md)
остается действующим, его «Осталось» перенесено в конец этого документа.
Уточнения сохраняются: **KateGB не проверять**; **приёмку реального проекта выполнять
только через серверное окружение**. Локальные проверки не являются приёмкой реального проекта.

## Сохранённые проверки

`2eeb877` — preflight вида представления при импорте и исправление Windows reserved device names.
Импорт репрезентации теперь получает явный `kind`: `import_source_suffixes(kind)` —
VECTOR → `{".cif"}`, IMAGE → `FRAME_IMAGE_SUFFIXES` (`.jpg`, `.jpeg`, `.bmp`, `.png`);
папка без подходящих источников или с чужим видом даёт явные `ImportIssue`
(`no_import_sources`, `ignored_files`), а не тихий импорт не того вида.
Каталоги workspace создаются и удаляются через `_directory_target`
(`_make_directory`/`_remove_tree` в `src/kraken_manager/infrastructure/workspace_files.py`),
поэтому зарезервированные DOS-имена (`aux`, `con`, …) на любой глубине больше не роняют
scan/create/import/delete.

Однорядочный WIP-дефект в `import_layer`
(`src/kraken_manager/infrastructure/workspace_files.py`):
`(stage / category).mkdir()` → `_make_directory(stage / category)` — дерево источника с
подкаталогом-зарезервированным именем (например `con`) роняло копирование aux (WinError 267).
Регрессия: `tests/test_workspace_files.py::test_import_layer_survives_reserved_aux_subdirectory`.

| Блок | Результат | Файлы |
|---|---|---|
| Preflight вида импорта + reserved device names (`2eeb877`) | 54 PASS, 21.51s; повторный прогон 2026-11-06 из этого checkpoint | `tests/test_workspace_files.py`, `tests/test_dataset_vector_inputs.py`, `tests/test_catalog_operation_parity.py`, `tests/test_hub_catalog_route.py`, `tests/test_hub_embedded_service_parts/`, `tests/test_server_workspace_inputs_parts/`, `tests/test_work_orders.py` |

## Окружение 2026-11-06

Пробы выполнены 2026-11-06 из `C:\code\kraken` (PowerShell 5.1, `.venv`).

- PostgreSQL: `Get-Process postgres` — **9 процессов**; локальный PostgreSQL жив.
- kraken-server НЕ запущен: `Invoke-WebRequest http://127.0.0.1:8080/health` —
  `System.Net.Sockets.SocketException: No connection could be made because the target machine
  actively refused it 127.0.0.1:8080`; TCP-слушателя на 8080 нет.
- Gateway НЕ запущен: `http://127.0.0.1:8081/health` — тот же отказ
  (`...actively refused it 127.0.0.1:8081`); TCP-слушателя на 8081 нет.
- `C:\kraken_test` существует: каталоги `blobs`, `source`, `derived` (плюс `.blob-staging`, `_rt`).
- `D:\Heggs\Source` — **только-чтение исходные данные**, изменения запрещены. 11 верхних
  каталогов-слоёв: `Heggs_2W1_FS`, `Heggs_3W2a_FS`, `Heggs_3W2_FS`, `Heggs_I`, `Heggs_II`,
  `Heggs_IPoli_SiW1`, `Heggs_M1_FS`, `Heggs_M2_FS`, `Heggs_M3_FS`, `Heggs_M3_FS_img 239-240`,
  `Heggs_Poli_Si_FS`. В каждом слое — подкаталоги `<слой>_img` и `<слой>_pls` (иногда `.psd`).
  Пробы трёх слоёв: `Heggs_M1_FS_img` — 324 изображения (`.jpg`) из 973 файлов,
  `Heggs_3W2_FS_img` — 324 из 972, `Heggs_I_img` — 324 из 972. Нумерация кадров линейная,
  `000..323`, т.е. **18×18 = 324 кадра на слой**, как указано пользователем.
  `Heggs_M3_FS_img 239-240` — отдельная верхняя папка с подмножеством кадров 239/240
  слоя `Heggs_M3_FS`.
- `kraken-admin doctor`: `.venv\Scripts\python.exe -m kraken_admin doctor` — exit 2,
  `[ERROR] configuration: [WinError 2] Не удается найти указанный файл:
  'C:\Users\shust*\AppData\Local\Kraken\LocalServer\server.toml'`. Ожидается:
  путь по умолчанию на этой машине не существует; запуск идёт через
  `kraken-server --config config/local/server.toml`. Никуда `server.toml` не создавался
  и не копировался.
- `config/local/`: `server.toml` (2470 B), `database-url.secret` (293 B),
  `blob-gateway.secret` (277 B). Все три игнорируются `.gitignore` строкой 39
  (`config/local/`): `git check-ignore` подтверждает каждый файл, `git status` по
  `config/local/` пуст. Секреты в git не попадают.

## Осталось (перенос из checkpoint 2026-11-01)

1. Отсутствует эталонная `0518.cif`: одна пропущенная регрессия Contour (skip из-за
   отсутствующей фикстуры, не подменена). После получения файла выполнить только её;
   остальные 324 проверки повторять без причины не нужно.
2. Legacy confidence: headless-запуск выдаёт только счётчик (operation/job_id/input_count);
   метрики PARTIAL. Не выдавать его JSON за оценку модели.
3. BLOCKED: Windows service (нужен отдельный elevated стенд, `Administrator=False` вне
   sandbox); Hello (нужно интерактивное подтверждение пользователя); updater/remote HTTPS
   (нужны тестовый установщик и выделенный TLS endpoint).
4. Живая приёмка реального проекта через серверное окружение:
   - Contour через machine worker — уже PASS в `393bca4` (строка матрицы
     LIVE-CONTOUR-MACHINE-WORKER).
   - Отказы live worker — уже PASS в `95ec8f2` (строка матрицы LIVE-WORKER-FAULTS).
   - NeuralImage (`frames.binary-segment.v2`, переиспользование обученного чекпоинта)
     и Karakal (`karakal.analyze.v1` через настоящий диалог Hub) закоммичены **после**
     checkpoint 2026-11-01 (его последнее обновление — `e9aca87`) в `2428a38` как live-тесты
     `tests/test_live_server_plugin_chain.py` с env-гейтом `KRAKEN_LIVE_WORKER_CONFIG`;
     сшивка (`layer.stitch.v1`, настоящий файл-диалог, machine compute, повторное чтение
     сохранённой мозаики без пересчёта) — в `1cf5539`, плюс `388dfab` (reopen мозаики в
     диалоге). В диапазоне `git log --oneline 9dd222a..HEAD` только план `4992aa7` и
     WIP-фикс `2eeb877` — новых приёмочных шагов в этом диапазоне нет.
   - Не выполнено: пользовательский GUI-запуск (настоящий `kraken-hub`, вход, MyWork,
     accept, история) и финальные шаги: revoke всех machine tokens, сравнение
     protected-базлайна, обновление матрицы. Строка матрицы REAL-PROJECT-SERVER-ACCEPTANCE
     ещё PARTIAL и не отражает `2428a38`/`1cf5539` — её нужно обновить в WP4.
   - Качество модели и электрическая приёмка реального прибора — вне области (явная
     граница в каждой квитанции).

Стабилизация не завершена, Kraken готовым не объявляется.

## Точный следующий шаг

Task 1.1 плана: машиночитаемые реестры — `docs/quality/registry/defects.json`,
`docs/quality/registry/actions.json`, `tools/quality/registry_schema.py` и
`tests/test_quality_registry.py`.
