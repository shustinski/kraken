# Актуальность повторных резервных копий

Проверяется функциональная готовность; обучение модели не запускалось.
Исходные JPEG/CIF из `D:\ozi\NN` не изменяются.

## Воспроизведение

`backup-reuse-observation/report.json` в
`.codex_tmp_scratch/production-audit-20261027/` зафиксировал дефект:
после создания нового поручения источник имел 9 событий, возвращённый архив —
8. Экспортёр безусловно возвращал любой ранее завершённый архив того же
проекта, а административная форма каждый раз использовала один каталог.
Старый архив оставался корректным историческим снимком, но объявлять его
новой копией текущего проекта было неверно.

## Контракт и исправление

- Каждый новый запрос из административной формы получает отдельный каталог
  с UUID. Завершённый исторический архив не перезаписывается.
- Экспортёр под блокировкой проекта сопоставляет текущий источник с готовым
  архивом: canonical event bytes/count/position, inventory blobs, SHA и пути
  snapshots, descriptor, external references и source profile.
- При совпадении `resume=True` возвращает прежний результат без изменений.
  При различии запрос отклоняется с указанием выбрать новый каталог.
  `resume=False` также запрещает reuse завершённого каталога.
- Незавершённый экспорт связан с SHA описания источника в checkpoint.
  После изменения источника он отклоняется до записи в старый каталог.
  Старые checkpoints без такой привязки также требуют нового каталога;
  завершённые v1 manifests сохраняют совместимость.
- Перед публикацией manifest повторно проверяются descriptor, snapshots и
  inventory. Legacy projection-only guards действуют и при повторном запросе.

Blob inventory использует идентификаторы SHA неизменяемого content-addressed
хранилища. Копируемые файлы сверяются с ожидаемыми SHA; завершённый архив
полностью проверяется перед reuse. Байты внешних ссылок, как и прежде,
не входят в архив. Для чтения исторического результата предназначены
`load_bundle_manifest` и `BundleVerifier`, а не новый запрос export.

## Доказательства

`backup-targeted.xml`: 10 top-level tests и 13 reported subtests прошли.
Проверены шесть независимых изменений источника, неизменность всех байтов
старого архива, restore нового события, отказ/resume после диагностического
сбоя, два настоящих Qt нажатия с переименованием проекта между ними,
восстановление последнего имени и обоих managed payloads в свежий каталог.
Также прошли существующие проверки legacy Qt migration и lifetime.

`backup-source-desktop/desktop-self-test-53da0c760c544cdc875756199d48d0ce/report.json`:
собственный профиль импортировал 16 реальных JPEG/CIF, выполнил Qt migration,
создал архив, изменил состояние поручения без запуска job, отказался вернуть
устаревший архив, создал новый и восстановил его в независимый каталог.
Replay восстановил именно изменённое состояние. SHA исходников совпали.
Поле `latest_backup_restored=true`; временная запись OS secret store очищена.

Изменённые шесть модулей прошли scoped Ruff.

`backup-root-final/summary.json`: 209 модулей, 1371 XML checks
(1340 top-level tests + 31 reported subtests), 0 failures/errors,
15 skipped; PostgreSQL включён, фактическое обучение отключено.
2684 pinned files не изменились во время проверки и сборки;
причины пропусков находятся в `backup-root-final/skips.json`.
Перекрывающиеся targeted/source gates к этому числу не прибавляются.

`backup-build-exit.json`: обе сборки завершились с exit 0 за 232.989 s.
`backup-desktop-sources.json` и `backup-server-sources.json`: соответственно
469 и 377 frozen project/Contour modules совпали с текущими исходниками.
Фактический bundled Desktop plugin catalog также совпал байт в байт.

`backup-desktop-installed/report.json`: фактические audit
install/acceptance/uninstall прошли за 20.812 s. Отдельный AppId
`KrakenDesktopAudit-6d050b4f364b43679364d9de629a7830`; регистрация и EXE
удалены, service ownership hash не изменился.
Installed self-test `desktop-self-test-564e13848605475bb481cf2a632a0633`
подтвердил `stale_backup_refused=true`, `latest_backup_restored=true`,
16 реальных пар и `source_files_unchanged=true`. Также прошли прежние
matrix/thumbnail/head refresh/reopen/settings, native SciPy/stitch/Agent gates.
Это внутренний Qt ввод в собственном диагностическом профиле.

`backup-server-installed/report.json`: audit install/acceptance/uninstall
прошли за 17.724 s. AppId
`KrakenServerAudit-ce19608a02fc47d7876756fe1e2c3e9a` удалён вместе с EXE;
service ownership hash не изменился. Собственная UUID schema PostgreSQL
удалена. Frozen Server/Admin/Gateway выполнили migrations/bootstrap/auth,
upload/Range, legacy import/retry, SciPy extraction и graceful process
restart с сохранением legacy/electrical state. Это file install и процессы
в контексте пользователя, не SCM service installation.

`backup-root-final/post-install-source-check.json` подтвердил неизменность
всех 2684 pinned files после обеих установок. Обучение не запускалось.

Production installers скомпилированы в `backup-installers/production/`,
не установлены и не опубликованы:

- Desktop: 74 537 548 bytes, SHA-256
  `a61b0c2ab91b78e2018fdf85bd23ee591f72f8dbd370a8e2c145422a3507943d`.
- Server: 114 882 680 bytes, SHA-256
  `2290357822fe6be130e4f3f9149a19b288e7dd7ab0852f7b539bc4389b991a6c`.

Audit installers собраны из тех же distribution directories с отдельными
AppId. Это не проверка production AppId update или побитовой воспроизводимости.

Это не подтверждает полный desktop/remote 20-stage E2E, восстановление после
отказа носителя или backup bytes внешних ссылок. Native Computer Use pipe
недоступен; обычный вход через пользовательские окна не засчитан.
