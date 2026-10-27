# Функциональная приёмка 26 октября 2026

По указанию пользователя текущий этап проверяет функциональную готовность.
Длительное обучение и оценка качества векторизации не запускаются.
Полная готовность системы пока не подтверждена.

`current-root-functional/`: 150 модулей, 1046 тестовых случаев, 0 failures,
0 errors, 13 skipped. Каждый модуль запущен отдельным процессом, PostgreSQL
проверен в одноразовых схемах собственного тестового контейнера. Пропуски
не засчитаны как успешные сценарии. Проверка ограниченного train/infer gate
в этом прогоне не включалась; используется ранее полученный checkpoint.

После обоих исправлений повторён **весь текущий корневой набор**:
`preserved-root-functional/` — 153 модуля, 1059 тестовых случаев, 0 failures,
0 errors, 13 skipped. Это 1046 выполненных случаев и 13 пропущенных, а не
1059 выполненных. POSIX fork проверен отдельно в Linux; opt-in training и
отдельные live UI/server gates в этом прогоне не включались.

`remote-analysis-acceptance/`: 5 тестов проверили Qt transport, отзывчивость,
CSV, повтор потерянного ответа с исходным idempotency key, уничтожение
родителя при активном потоке, действительные PostgreSQL/HTTP create/cancel,
ACL viewer/outsider/anonymous, CAS и изоляцию проектов. Реальный subprocess
Karakal проверен отдельно: [протокол](2026-10-26-server-analysis-functional.md).

## Найденный дефект восстановления модели

**GAP-B01 / BLOCKER / aggregate plugin import → backup restore.** Импорт
модели сохранял `representation_id` и `frame_id` как строки `"None"` для
layer attachment. Текущая проекция работала, но replay после восстановления
останавливался на UUID validation. Воспроизведено на ранее созданном настоящем
checkpoint, без повторного обучения. Ошибка сохранена в каталоге
`functional-backup-restore-02/model/bundle/`.

Исправлены `application/plugin_jobs_parts/call.py` (JSON null для новых событий)
и `infrastructure/projections/rebuild.py` (чтение точного исторического `"None"`
только для attachment scope). Журнал и исходные blobs не переписываются.
Некорректные UUID и такое кодирование в frame scope по-прежнему отвергаются.
`test_plugin_attachment_replay.py`: 3 passed, включая действительный обработчик
импорта; соседние plugin application/migration модули также прошли.

Повторная проверка `functional-backup-restore-03/receipt.json`: восстановлены
два проекта. Модель: 37 событий, 6 managed versions, dataset/model/mask SHA и
lineage, завершённый pipeline после удаления projection cache и reopen.
Принятая техническая CIF-правка: 17 событий, 3 версии, accepted state и полная
история исправления/независимой проверки. Журналы исходных проектов неизменны.
Оба проекта прошли scan_integrity. GAP-B01 исправлен и перепроверен в исходниках;
проверка обновлённых frozen пакетов фиксируется отдельно ниже.

Повторяемый инструмент: `tools/verify_functional_backups.py --model-project
<completed-model-gate> --quality-project <completed-quality-gate> --output
<fresh-directory>`. Проверяет проекты, а не перенос аккаунтов/секретов или
PostgreSQL disaster recovery.

## Упаковка и открытые проверки

**GAP-D01 / CRITICAL / remote download cleanup.** Оба пути скачивания
(`download`, `download_url`) вызывали unlink даже при HTTP-ошибке до создания
файла или при FileExistsError. Это удаляло независимый существующий файл.
В `download-preservation-before/` воспроизведено 6 failed / 2 passed.
Исправление в `remote_client_parts/download.py` запоминает identity только
после успешного exclusive create и очищает только совпавший файл. После
исправления 8 passed: collision, HTTP failure, partial stream/retry и замена
другим писателем для обоих путей. Соседние remote catalog/work package/asset
модули также прошли. Исправление не меняет содержимое существующих файлов.

Первые актуальные `functional-installers/` прошли действительную установку,
self-test и удаление: Desktop 9 проверок за 9.246 s, Server 12 за 18.788 s.
Но эти бинарники собраны **до** исправления GAP-B01 и не доказывают его наличие
в поставке. `restorable-build` содержит GAP-B01, но ещё не GAP-D01.
Оба исправления собраны в `preserved-build`. Проверены встроенные модули
Desktop (import, replay, download, remote analysis) и Server (import, replay)
на соответствие исходникам: `preserved-desktop-sources.json` и
`preserved-server-sources.json`.

Итоговый Desktop audit-установщик (`preserved-desktop-installer/report.json`)
прошёл install, 9 self-test проверок и uninstall за 8.390 s; регистрация и
executable удалены. Итоговый Server audit-установщик
(`preserved-server-installer/report.json`) прошёл install, 12 проверок
миграций/auth/HTTP/blob/extraction/restart и uninstall за 17.762 s;
регистрация и executable удалены, служба отсутствовала до и после проверки.

`tools/verify_frozen_sources.py` сравнивает встроенные code objects выбранных
модулей с текущими исходниками (различается только путь исходного файла).
Проверка подтвердила GAP-B01 в `restorable-build` и правильно отвергла старый
download module этой же сборки. Это проверка содержимого пакета, а не замена
функциональному запуску установленного приложения.

Проверки используют свежие audit AppId и каталоги; производственная служба
не создаётся. Текущий токен не имеет SCM create-service права (WinError 5),
поэтому реальная service registration/reboot и clean OS остаются непроверенными.
На момент этих установочных проверок были открыты удалённая карта расхождений, actual runtime attestation worker,
единый 20-шаговый межподсистемный сценарий и нагрузочные ограничения.
Качество алгоритмов не подменяется прохождением этих функциональных проверок.

Позднее удалённая карта реализована и проверена в исходниках:
[протокол карты](2026-10-26-analysis-map-functional.md). Эти установщики её ещё
не содержат. Локальный настоящий рендер позднее также реализован:
[локальная карта и отмена](2026-10-26-local-analysis-map-functional.md).
Последний root gate — 156 модулей, 1076 test cases, 0 failures/errors,
13 skipped. Runtime verification позднее реализована и проверена:
[среда worker](2026-10-26-worker-runtime-functional.md).
Общий E2E, вычисляемая сшивка, нагрузка и финальная упаковка открыты.

Все указанные каталоги находятся под
`.codex_tmp_scratch/production-audit-20261026/`.

Последующие проверки: `worker-runtime-post-audit-native/` — 21 cases passed
после независимых исправлений fingerprint; `remote-cache-digest/` — 19 cases
passed, включая повреждение cached file при неизменном размере. Remote cache
теперь сверяется по SHA-256 перед повторным использованием.

Для GAP-STITCH01 реализовано и проверено промежуточное вычислительное ядро,
versioned physical geometry и local/HTTP/PG commands:
[состояние сшивки](2026-10-26-computed-stitching-gap.md). Полный pipeline/UI
workflow с публикацией mosaic ещё не готов; gap не закрыт. Длительное обучение
не запускалось, качество векторизации в этих gates не оценивалось.

Последний общий корневой gate: `stitch-foundation-root-functional/`,
162 модуля, 1113 cases, 0 failures/errors, 13 skipped (1100 выполнены).
Он включает runtime, remote cache SHA, основу вычисляемой сшивки и прежние
local/server/PG/Qt сценарии. Незаконченный Agent/Pipeline/UI workflow
сшивки этим не объявляется готовым. Дистрибутив после этих изменений
ещё не пересобран и не прошёл новую установленную приёмку.

Позднее реализован локальный полный slice сшивки: Qt -> built-in Agent ->
native worker -> indexed mosaic artifact -> atomic local import -> block
viewer; отдельно проверены atomic PG adapter и canonical backup/restore:
[протокол локальной сшивки](2026-10-26-local-stitch-functional.md).
Последний root gate `stitch-local-root-functional/`: 165 модулей, 1131 cases,
0 failures/errors, 13 skipped (1118 выполнены). Server queue/remote UI и общий
typed layer Pipeline ещё не подключены. Большие нагрузки и final installed
gate остаются открыты; полная функциональная готовность не объявляется.

Следующий этап подключил сшивку к существующей server Agent queue, V2 input
transfer, native subprocess, publication/import и lease/cancel/recovery:
[серверный протокол](2026-10-26-server-stitch-functional.md).
Последний root gate `stitch-server-root-functional/`: 166 модулей,
1138 cases, 0 failures/errors, 13 skipped (1125 выполнены). Проверен также
отказ после деактивации инициатора с program attribution и отсутствием
output. Remote UI сшивки, общий layer Pipeline, нагрузка и финальный
установленный дистрибутив ещё не приняты; цель остаётся незавершённой.

Remote Qt сшивка позднее реализована и проверена: JSON observations ->
server Agent/native result -> bounded block/coverage API -> real Qt viewer,
без local Agent и full mosaic download. Сохранены retries с прежним ключом,
исправлен UI deadlock после validation rejection, сохранена compatibility
старого mosaic SHA: [remote UI протокол](2026-10-26-remote-stitch-ui-functional.md).
Последний root gate `stitch-remote-root-functional/`: 167 модулей, 1143 cases,
0 failures/errors, 13 skipped (1130 выполнены). Общий layer Pipeline,
vector geometry, большие нагрузки и final installed gate ещё открыты.

Позднее в общий граф добавлены typed layer nodes с отдельными geometry/content
bindings: [протокол layer Pipeline](2026-10-26-layer-pipeline-stitch-functional.md).
Пройдены реальные native local и PostgreSQL/HTTP execution/import, mixed
inference→mask→mosaic на явно необученном checkpoint (обучения нет), Qt RU/EN
сохранение настроек и desktop background continuation. Проверены concurrent
PostgreSQL reservation, transaction rollback, глобальная pause/cancel recovery,
late quality rejection и explicit retry с новым attempt. Canonical backup и
rebuild mixed проекта сохранили 38 событий, workflow и mosaic SHA/coverage.
Это растровый слой pipeline; vector/CIF stitching, большой E2E, нагрузка и
актуальный установленный дистрибутив остаются открытыми.

Последний root gate `stitch-pipeline-root-functional/`: 171 модуль,
1155 cases, 0 failures/errors, 13 skipped (1142 выполнены). Добавленный после
collection модуль отказов проверен отдельно: 3 cases passed в
`pipeline-stitch-failure-boundaries/`. Scoped gates с root не суммируются.
Обучение не запускалось. Полная эксплуатационная готовность по-прежнему
не подтверждена; установщики пока не содержат последние изменения.

Следующий этап добавил native vector content в тот же layer Pipeline:
[протокол векторной сшивки](2026-10-26-vector-stitch-functional.md).
Настоящие local и PostgreSQL/HTTP Contour→CIF→vector mosaic/import прошли;
Qt сохраняет vector edge и показывает расхождения; backup/rebuild сохранён.
Исправлена JSON-сериализация nested immutable parameters в ServerAgentWorker.
Полный read-only проход реального корпуса: 5498 пар, 4944 прочитаны,
554 отказа из-за формата/неоднозначной топологии. Он **не считается passing
gate** и не доказывает готовность всего корпуса. Исходники неизменны.
Обучение и оценка качества векторизации не запускались. Сложный CIF,
экспорт/дальнейшее потребление layer vectors, нагрузка и final installed gate
ещё открыты.

Последний общий gate после vector интеграции:
`stitch-vector-root-functional/summary.json`, 176 модулей, 1168 cases,
0 failures/errors, 13 skipped (1155 выполнены). Он включает новые native
local/server/Qt vector сценарии. Отдельный real corpus gate с 554 отказами
не засчитывается в passing результаты. Установщики остаются прежними,
обучение не запускалось, полная готовность не объявляется.

Далее реализованы export/re-read vector mosaic с опорными и SI координатами,
provenance/conflicts и coverage/no-data через local/remote Qt и CLI:
[экспорт и диагностика](2026-10-26-vector-exchange-functional.md).
Native local/server export, network interruption/GUI retry, no-overwrite,
complete seal и Windows→Linux→Windows byte-identical round trip прошли.
Исправлены silent omission вырожденного P и Windows progress sharing;
ошибки CIF содержат frame/version/polygon/command. Полный повторный corpus
parse — 4944/5498 прочитаны, 554 явных отказа; исходники неизменны.
Последний root `stitch-vector-exchange-coverage-root/`: 177 модулей,
1184 cases, 0 failures/errors, 13 skipped (1171 выполнены). Установщики
не пересобраны, downstream IR/электрическая приёмка и общий E2E открыты.
