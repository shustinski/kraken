# Фактическая среда Karakal worker

Длительное обучение и оценка качества векторизации не выполнялись. Численные
проверки используют контролируемые маски и настоящий Karakal Worker.

Новые AnalysisRun partitions требуют `worker_runtime`. Запись формируется
в вычисляющем процессе: version.py Karakal, Python, реально загруженные
NumPy/OpenCV, ОС и SHA-256 загруженного Python-кода вычисления/контракта.
Fingerprint охватывает нормализованный loaded bytecode функций, методов и
property accessors семи явно перечисленных в `runtime_identity.py` модулей,
поддерживаемые default/keyword-default значения и module constants. Значения
set/frozenset кодируются канонически; неподдерживаемые объекты обозначаются
только типом, без address-bearing repr. Это не хеш произвольного состояния
closures/импортированных объектов или всех Python-модулей.
Пути co_filename исключены из fingerprint. Проверены независимые процессы,
разные PYTHONHASHSEED и изменение загруженной функции. JSON metadata Karakal
согласована с package/version.py: 1.0.0 вместо прежнего 0.0.1.

Это запись доверенного worker, привязанная к аутентифицированному Agent/job/
partition и сохранённая в каноническом результате. Это **не** аппаратная
аттестация и не доказательство честности злонамеренного worker. Fingerprint
не является хешем всех нативных DLL, CPU или драйверов. Такие утверждения
этой проверкой не делаются.

## Поведение и совместимость

- Первый принятый worker runtime закрепляется за анализом. Другой runtime
  того же run не импортирует свои frame records: FileStore использует project
  lock, PostgreSQL — run row lock и канонический event CAS.
- Когда expected_runtime задан явно, настоящий worker отказывает до чтения
  изображений и вычисления metrics. При неизвестном runtime начальные
  одновременно запущенные partitions проверяются при импорте; автоматический
  подбор совместимых узлов очередью пока не реализован.
- Retry/Repeat сохраняют ограничение expected_runtime; новая карта закреплена
  за принятой средой исходного анализа. Отказ сохраняет причину и фактическую
  запись среды без вымышленных metrics.
- Сравнение и independent quality recheck используют worker receipt вместо
  предположения Hub/Server о своей локальной среде. Изменение declared host
  metadata при той же измеренной среде не подменяет идентичность worker.
- Старые JSON без receipt остаются читаемыми и восстанавливаемыми как
  unverified. Они не доказывают сопоставимость и не допускаются к автоматической
  quality evaluation. Нужен новый запуск и измеренный baseline; старые записи
  и решения не переписываются и не объявляются автоматически эквивалентными.
- В панелях local/remote доступны runtime tooltip, окно «Среда worker» с
  expected/reported/accepted версиями и ошибкой отказа; viewer карты также
  показывает сохранённый receipt.

Исправлена дополнительная ошибка Agent: GET native result вызывал отсутствующий
`to_dict`. Теперь общий сериализатор обслуживает V1/V2/native results. Сообщение
нативного отказа сохраняется в Agent job.error и в AnalysisRun history.

## Evidence

Все каталоги под `.codex_tmp_scratch/production-audit-20261026/`:

- `worker-runtime-repeat-and-ui/`: 6 модулей, 26 passed, 0 skipped. Реальный
  local Agent/subprocess и server HTTP/PG, карты/Qt, ожидаемая среда при Repeat.
- `worker-runtime-pg-negative/`: 4 модуля, 8 passed. Настоящий server worker
  отказывает несовместимому запросу, receipt переживает reopen/replay; реальные
  PostgreSQL rows не смешивают две разные среды и не теряют первый partition.
- `worker-runtime-negative/`: 4 модуля, 31 passed, включая отсутствие receipt,
  наследуемый формат, первый/второй partition, loaded code и качество recheck.
- `worker-runtime-diagnostics-ui/`: 6 модулей, 54 passed. Окно runtime открывается
  через настоящие local/server Qt controllers, включая local failure receipt.
- `worker-runtime-host-declaration/`: 10 passed. Включён положительный случай
  одинакового worker при разных declared host metadata и отказ legacy baseline.
- `runtime-root-functional/`: 158 модулей, 1088 test cases, 0 failures/errors,
  13 skipped (1075 выполненных случаев). Этот root gate предшествует добавлению
  runtime details button; кнопка проверена отдельным UI gate выше. Прогоны
  пересекаются и не суммируются.

Независимый read-only аудит после этих gates выявил и позволил исправить
пропуски property accessors, `__defaults__`, `__kwdefaults__`, frozenset constants
и перенос learned runtime pin при Retry (local/server). Поведенческие fault
fixtures подтверждают, что изменения теперь меняют fingerprint; разные
PYTHONHASHSEED сохраняют одинаковый hash. Local retry fixture использует 1001
frame record, server retry проверен настоящим subprocess после corrupt-input
attempt. Evidence: `worker-runtime-independent-audit-fixes/`,
`worker-runtime-audit-defaults/`, `worker-runtime-audit-retry-server/`.
Mixed legacy receipts и отсутствие одного cached receipt не объявляются
полностью измеренными: `worker-runtime-mixed-legacy/`,
`worker-runtime-pg-missing-mixed-final/`.

Повторный scoped read-only аудит подтвердил устранение перечисленных omissions:
13 runtime receipt tests passed; `worker-runtime-fingerprint-final/` — 3 модуля,
20 tests passed, 0 failures/errors/skips. Это независимая промежуточная проверка
runtime scope, не финальная независимая приёмка всего Kraken.

Runtime fault fixtures помечены как metadata/control fixtures и не выдаются
за второй реально установленный набор библиотек. Реальная среда текущего
worker отдельно проверена в subprocess и через HTTP. Новая функция пока
не проверена в финальном frozen-дистрибутиве. `preserved-installers` устарели.
Общий 20-шаговый E2E, вычисляемая сшивка, нагрузка, service/clean OS и финальная
упаковка остаются открытыми; полная производственная готовность не объявляется.

После финальных исправлений fingerprint повторён native local/server/PG gate:
`worker-runtime-post-audit-native/` — 5 модулей, 21 test cases,
0 failures/errors/skips. Выполнены реальные subprocess/HTTP AnalysisRun и
quality recheck; обучение модели в этом gate не запускается.
