# Локальная карта анализа и восстановление отмены

Длительное обучение и оценка качества векторизации не выполнялись. Проверка
использует настоящий Karakal Worker на контролируемых бинарных масках.

Локальная кнопка карты больше не подменяет изображение сообщением с числами.
`LocalAnalysisMapController` передаёт запрос существующему AnalysisRunCoordinator
и локальному Agent. Используются исходные версии, SHA, recipe и parameters
выбранного кадра. Нельзя подменить входы/координаты/параметры карты. Готовая
карта повторно показывается из BlobStore, а не пересчитывается.

Намерение, результат, отмена и подтверждение импорта хранятся как
`analysis_map.*` события в существующем `analysis-run:<uuid>` stream. Состояние
карты читается из этих событий; отдельной БД, очереди или авторитетной таблицы
нет. PNG сохраняется в существующем project BlobStore. Метрики исходного
анализа не меняются. Импорт проверяет SHA/размер и общие PNG allocation limits.

GUI выполняет сетевые и файловые операции в сохранённых фоновых потоках,
отдельно переживающих удаление родительского QWidget. Закрытие Desktop видит
эти потоки через существующий background-workers guard. Cancel/retry/repeat
локальной панели также перенесены в фон. Один общий viewer используется для
локальных и удалённых карт: немодальное окно, scroll, выражение, легенда
операндов и раскрываемые immutable input versions/SHA.

Проверено:

- настоящий локальный HTTP Agent → subprocess Karakal → PNG → BlobStore;
- сбой submit после сохранения intent, rebuild и доставка того же job ID;
- сбой acknowledgement после сохранения PNG, повтор после перезапуска и
  переход Agent job в succeeded без повторного вычисления;
- удаление analysis SQLite cache и временного PNG Agent не теряет карту;
- canonical backup/import/rebuild восстанавливает карту и прежние metrics;
- actual DesktopController `_configure_analysis` → кнопка панели → настоящее
  немодальное окно с пикселями рассчитанной карты и input-version details;
- responsiveness при блокированном RPC и удаление родителя при активном потоке;
- отмена/поздний результат, подмена входов и параметров отвергаются.

Evidence в `.codex_tmp_scratch/production-audit-20261026/`:
`local-analysis-map-backup-final/` — 6 модулей, 19 passed, 0 skipped;
`local-map-desktop-composition/` — 2 модуля, 6 passed. Qt transport-only
fixtures не считаются проверкой алгоритма; native subprocess gate отделён.
Прогоны пересекаются, количества не суммируются. Полный текущий root gate
сохраняется отдельно в `local-maps-root-functional/`.

Этот полный root gate завершён: **156 модулей, 1076 test cases, 0 failures,
0 errors, 13 skipped** (1063 выполненных случая). Пропуски не считаются
успешными проверками; opt-in обучение не включалось. После уточнения теста
Desktop composition он повторно прошёл отдельным прогоном, указанным выше.

## Дополнительный дефект отмены

**GAP-A-CANCEL01 / MAJOR / local AnalysisRun.** Ранее cancel сначала вызывал
Agent, затем записывал отмену. При сетевом сбое авторитетной отмены не было;
поздний result мог импортироваться. Теперь cancellation intent записывается
до сети и повторно доставляется при refresh. Import и state changes
сериализованы существующим project lock; terminal cancelled/failed attempt
не может принять поздний result. `test_analysis_dispatch_recovery.py`
воспроизводит offline cancellation, replay, доставку после восстановления
и отказ позднему импорту. Это не утверждение о power-loss/SCM/reboot тестах.

Измеренная среда worker позднее реализована и проверена:
[протокол runtime](2026-10-26-worker-runtime-functional.md).
Открыто: общий 20-шаговый E2E,
нагрузочная приёмка и финальная упаковка. `preserved-installers` созданы до
обеих карт и их не содержат. Общая производственная готовность не объявляется.
