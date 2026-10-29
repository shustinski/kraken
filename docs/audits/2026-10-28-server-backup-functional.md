# Согласованное восстановление Kraken Server

Этап функциональной готовности; обучение моделей не запускалось.
Инструкция оператора: [backup/recovery](../server-backup-recovery.md).
Артефакты: `.codex_tmp_scratch/production-audit-20261028/`.

## Реализованный протокол

Добавлены локальные команды `backup-server`, `backup-verify`, `restore-server`.
Архив содержит PostgreSQL custom dump, три настроенных хранилища, SHA/размеры,
пустые каталоги, alias вложенных/совпадающих корней, инвентарь таблиц и
несекретные параметры конфигурации. Exported MVCC snapshot применяется и к
инвентарю, и к `pg_dump`. До snapshot удерживается общий PostgreSQL storage
barrier. Прямые файловые команды локальных сервисов также учитывают барьер.
Удаление staged trash и фиксация progress теперь находятся под этим барьером.

Python и Rust Gateway используют общий kernel file gate: Python exclusive,
HTTP upload/download shared до завершения body. Запуск второго Gateway
не выполняет очистку staging, если другой процесс удерживает gate. Staging,
transfer leases, gate и `_kraken_recovery` исключаются из копии; исходные
временные файлы не удаляются. Существующие bytes проверяются до и после
копирования; отсутствие обязательного managed blob прерывает export.

Restore принимает только свежий каталог и пустую базу. Архив и pairing
проверяются до изменения целевого состояния. SQL и PostgreSQL receipt
фиксируются одной транзакцией; `--resume` распознаёт rollback и committed
receipt. Существующие несовпадающие целевые файлы сохраняются с отказом.
Конфигурация публикуется отдельно, URL БД шифруется заново, Gateway secret
меняется, ключ подписи review сохраняется. Старые event/provenance paths
не переписываются: runtime alias разрешает их внутри восстановленных корней.
Повторное восстановление сохраняет цепочку прежних корней.

## Найденные дефекты и регрессии

- `setup-server` падал с `UnboundLocalError`: локальный импорт в другой ветке
  dispatcher затенял `run_migrations`. Удалён дублирующий импорт; регрессия
  проверяет последовательность migrations/config/bootstrap.
- После переноса workspace `prepare-inputs` возвращал 422 из-за сравнения
  старого canonical source path с новым физическим каталогом. Сравнение и
  проверка принадлежащего проекту work destination используют root resolver.
  Native PG + authenticated HTTP regression теперь получает prepared_frames=1.
- Незавершённое удаление сохраняло исходные абсолютные пути. План текущей
  операции и шаги staging разрешаются в новые корни. Регрессия прерывает
  реальное удаление после staging, восстанавливает сервер и завершает cleanup:
  удаляется только восстановленная копия; исходный trash и failed record целы.
- Реальный frozen импорт JPEG/CIF выявил потерю artifact `parameters` при
  формировании Gateway transfer payload. Сохранение версии было успешным,
  но матрицы пустовали. `gateway-parameters-before.xml` воспроизводит потерю.
  Теперь параметры сохраняются, ограничены 64 KiB finite JSON object и
  привязаны SHA к подписанному билету. Изменённые параметры дают 401,
  некорректные — 422 до preflight. Пустые параметры сохраняют прежний ticket
  context. Новая frozen приёмка проверяет 16 непустых ячеек обоих представлений.
- Длинные Windows paths штатных PG tools нельзя было надёжно передавать через
  `--file`: pg_restore отказывал обычному длинному пути; psql превращал
  расширенный `\\?\` в некорректный `//?/`. Dump и SQL теперь передаются
  через открытые Python stdin/stdout handles. Scoped изолированные прогоны
  с длинными temp paths проходят без сокращения тестовых путей.

## Проверки

`tests/test_server_backup.py`: 24 native PostgreSQL cases. Используются
только собственные UUID базы; public schema реально мигрируется. Проверены
snapshot с поздним commit, все корни/пустые папки, повтор restore, rollback и
discarded reply после успешного psql, повреждённый manifest/bytes/traversal,
alias и case collisions, чужие table/function и изменённые целевые bytes,
изменение source во время export, повторная recovery chain, pending cleanup,
direct-command barrier, transient files, новый protected config и HTTP вход.
Discarded reply является внедрённым исключением после успешного subprocess,
а не физическим обрывом TCP в момент commit.

Четыре root relocation cases проверяют Windows paths/case, вложенные корни,
идемпотентность resolver и обязательность metadata. Два настоящих Rust HTTP
теста проверяют Python exclusive → PUT/GET 503, возврат успешной передачи
после освобождения, удерживаемый upload → отказ Python exclusive, а также
сохранение искусственно состаренного активного staging вторым Gateway.
Три штатных Rust unit tests прошли после актуальной сборки.

PG tools 17.11 взяты из официального EDB Windows ZIP; SHA ZIP совпал с
публикуемым `.sha256`: `80379b2c04d51c30225532e0ae04509899141e9957ed096fe749d7fd9df8f82f`.
Извлечены только клиентские bin/DLL; сервер и служба PostgreSQL не устанавливались.
Приёмка использует прежний собственный PG17 на loopback 58854.

`server-backup-frozen-real-user/report.json`: current Server/Admin/Gateway,
61.509 s, passed. В собственную новую базу через HTTP/Gateway загружены
16 разрешённых JPEG/CIF пар из `D:\ozi\NN`, созданы image/vector representations,
проверены матрицы, SHA полных файлов и Range. Actual EXE backup/verify/restore
и повтор `--resume` прошли. Восстановлены все три корня; review key сохранился,
Gateway key изменился. Исходная учётная запись, матрицы и все 32 файла
проверены после восстановления и ещё одного restart. Свои базы удалены,
SHA всех оригиналов совпали. Раскладка 4×4 — объявленная каталожная fixture,
не доказательство физического соседства или качества топологии.

Промежуточные ошибки самого verifier (повторный вход в открытый httpx client,
неполная fixture metadata) сохранены в первых frozen-real отчётах. После
исправления fixture подтверждён отдельный продуктовый дефект потери parameters.
В restricted frozen прогоне зафиксирован WinError 5 при финальной замене
filesystem receipt после восстановления SQL. Та же неизменённая EXE прошла
под обычной Windows учётной записью; причина единичного отказа не установлена.
Он не объявлен исправленным продуктовым дефектом.

## Сборки

`server-backup-build-exit.json`: Server и Desktop, exit 0, 209.56 s.
После parameters fix Server/Admin пересобраны отдельно:
`server-backup-current-build-exit.json`, exit 0, 113.332 s.
Все project modules в PYZ сверены с source: Desktop 472, Server 380,
Admin 380. Desktop не содержит изменённых server modules и повторной сборки
после metadata fix не требует.

`server-backup-desktop-installed/report.json`: 24.23 s, passed,
AppId `KrakenDesktopAudit-eb822508640e48e6aefdeee15fc6fa99`.
Actual install/self-test/uninstall сохранил прежние 16-pair matrix/head/reopen,
legacy/freshness, Qt administration backup/scan/restore/heartbeat/busy-close,
native SciPy/stitch и Agent gates. Settings изолированы, SHA оригиналов целы,
собственный OS credential очищен. Регистрация и EXE тестовой установки удалены,
service ownership hash не изменился. Dialog path/message answers остаются
внутренними fixtures; ordinary native UI/login не заявляется.

`server-backup-server-installed/report.json`: 81.052 s, passed,
AppId `KrakenServerAudit-82fde6a1f52c48c4bf8b12ca86a69b3d`.
Actual audit install/acceptance/uninstall выполнил прежние migrations/auth/
Gateway/Range/SciPy/legacy-import/restart gates и новый backup gate на тех же
установленных Server/Admin/Gateway. Вложенный `server-backup-acceptance`:
60.439 s, все 16 пар, две матрицы и 32 SHA/Range после restore/restart прошли,
собственные базы удалены, оригиналы не изменились. Регистрация и EXE audit
установки удалены; service ownership до/после null (служба не создавалась).

`server-backup-current-root/summary.json`: 213 изолированных модулей,
1415 XML checks (1384 top-level + 31 reported subtests), 0 failures/errors,
15 skipped, 1400 выполнены. PostgreSQL и native PG tools включены; обучение
выключено. `skips.json` сохраняет причины пропусков. Все 2697 pinned files,
включая Rust source/Cargo и новые Python modules/verifier, совпали с SHA до
прогона и после сборки/установок (`post-install-source-check.json`).
Первый root до metadata fix тоже прошёл, но актуальным является второй;
его результаты не суммируются с предыдущими или installed gates.

Production installers скомпилированы, не установлены и не опубликованы:

- Desktop: `server-backup-installers/production/KrakenDesktopSetup-0.1.0.exe`,
  74 558 693 bytes, SHA `7431f12c949c4c3e009e118a6a88d915b8977612281a5f7d4b2dcd8f62f8695a`;
- Server: `server-backup-current-installers/production/KrakenServerSetup-0.1.0.exe`,
  114 981 943 bytes, SHA `91e484ec36e414f3304a18006ab427d344935cbfca48402038e4c4cdcd298c6f`.

## Открытые границы

Это функциональный backup/restore gate текущего стека с отдельной базой и
своими каталогами. Внешние workspace paths/URI, PKI и программы вне трёх
корней остаются зависимостями оператора. Не проверены cluster roles,
PITR/WAL, tablespaces/extensions portability, многопользовательская длительная
нагрузка, физический диск/питание, SCM/clean OS/reboot и production AppId update.
Computer Use pipe остаётся недоступен; ordinary installed native login и полный
20-stage installed workflow не подтверждены. Причина исходного python.exe
access violation со снимка пользователя не установлена. Полная производственная
готовность всех §§1–19 и качество векторизации этим этапом не объявляются.
