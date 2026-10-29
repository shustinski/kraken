# Изоляция обычного GUI-профиля и сохранённых данных входа

Продолжение [проверки непустого installed UI](2026-10-27-nonempty-installed-ui.md).
Цель обычного запуска/login/process restart остаётся отдельной от self-test.
Обучение и оценка качества векторизации не запускаются.

## Обнаруженные проблемы

`windows_credentials.py` использовал одно имя Credential Manager
`Kraken.ProjectManager.LocalAccount` для всех каталогов данных. Вход в отдельный
audit/demo профиль мог заменить запомненный логин рабочего профиля. До исправления
такой обычный audit-вход не выполнялся.

Эксперимент также показал, что `QSettings.setDefaultFormat(IniFormat)` и
`setPath` не перенаправляют используемый двухстрочный конструктор
`QSettings("Kraken", "KrakenHub")` в нашей сборке Qt на Windows. Начальный новый
тест остановился на проверке пути до записи настроек
(`profile-isolation-scoped.xml`, 1 failed, 37 passed). Прежняя self-test настройка
формата не была достаточным доказательством изоляции всех GUI settings; это
ограничение исправлено, а прежние утверждения не расширяются задним числом.

## Исправления

Сохранённый логин теперь привязан к каноническому каталогу данных. Стабильное
имя содержит SHA-256 пути; сам путь в target не помещается. Стандартный
`Path.home()/.kraken` сохраняет прежнее имя, как и прямые вызовы API без target.
Для явно отдельных профилей используется отдельный target. Legacy account DB,
пароли, разрешения и данные проектов не мигрируются и не удаляются. Первый вход
в старый нестандартный профиль может потребовать ручного ввода: чужой общий
credential не используется для автоматического заполнения.

Проверка Windows identity перед autofill сохранена. `credentials_available`,
последующее чтение и запись получают один и тот же target выбранного профиля;
при отмене Windows verification секрет не читается.

Новый opt-in `--settings-dir` размещает настройки обычного Hub/Manager в
отдельных INI-файлах. `kraken_core.settings.QSettings` выбирает явный файл для
organization/application и отключает fallback к глобальным настройкам.
Все соответствующие Hub/Manager imports переведены на эту обёртку. Без
явного каталога сохраняется прежний native backend. Самопроверка также
использует явный каталог и проверяет фактический `fileName` до работы с GUI;
installer verifier требует `settings_isolated=true`.

## Доказательства до обычного запуска

- Scoped tests: 43 passed, `profile-isolation-scoped-guard.xml`.
- Три отдельных процесса записали/прочитали настройки профилей A/B: URL A
  сохранился после перезапуска, B остался пустым. Файл каждого процесса находится
  внутри явно выбранного каталога. Проверяется и сохранность default/native пути
  при отсутствии override.
- `production-audit-20261027/profile-credential-os/report.json`: passed.
  Две настоящие UUID-записи Credential Manager записаны и прочитаны независимо,
  затем удалены. Guard нативных `CredReadW`/`CredWriteW` запрещал все посторонние
  targets, включая legacy target, до вызова Windows API. Пароли в отчёт не выведены.

Отдельный обычный профиль подготовлен в
`production-audit-20261027/ordinary-ui-audit/profile`: один тестовый аккаунт
`kraken-audit-ui` с пустым audit-only паролем, проект `GUI_audit_d3e42c14`,
16 управляемых JPEG/CIF из разрешённого NN. Это отдельные копии, не исходные
пользовательские каталоги. State/ожидаемые SHA/version ID сохранены в
`ordinary-ui-audit/state.json`. Пока это API-подготовка и fresh-service readback,
не успешный обычный GUI login или process restart.

## Общий и установленный gate

`production-audit-20261027/profile-final-root/summary.json`: passed,
202 модуля (root + CSliser), 1303 cases, 0 failures/errors, 14 skipped,
1289 выполнены, настоящий audit PostgreSQL включён. Все 2659 закреплённых
source/test/tool/spec/dependency файлов неизменны, в том числе после сборки и
установочных проверок. Пропуски остаются отдельными live/model/training/E2E,
POSIX и symlink gate; они не засчитываются как выполненные.

`profile-final-build/`: сборка Desktop/Server с кодом 0. Полная bytecode-сверка
модулей проекта/Contour прошла: Desktop 465, Server 375
(`profile-final-{desktop,server}-sources.json`). InnoSetup production/audit
variants скомпилированы с кодом 0. Production AppId не установлен и не опубликован.

| Production installer | Bytes | SHA-256 |
| --- | ---: | --- |
| KrakenDesktopSetup-0.1.0.exe | 74 518 354 | `43c0b4326a75c54c7aa0cbbc1b7c824ab3e49f312b0eed8610bb88b25f477f47` |
| KrakenServerSetup-0.1.0.exe | 114 854 355 | `1fb98a0fd7db198570a0815d3003f942ca3eb313b2e4f32787daf567252d022d` |

Файлы находятся в `profile-final-installers/production/`.

- `profile-final-desktop-installed/report.json`: passed, 15,865 s,
  16 настоящих JPEG/CIF, managed matrix/thumbnails/head refresh/reopen,
  неизменность исходников, native workers, Agent и панели. Сам packaged exe
  проверил `settings_isolated=true`. Независимый просмотр квитанции подтвердил
  реальный INI `settings_file` внутри собственного `desktop-self-test-*/settings/`.
- `profile-final-server-installed/report.json`: passed, 17,640 s,
  миграции/bootstrap/auth/Gateway/Range/SciPy и restart/persistence с настоящим
  изолированным PostgreSQL.

Обе собственные audit installation удалены, регистрации и exe отсутствуют,
существующая Windows-служба не изменена. Это current-user файловая установка
Server, не SCM/reboot acceptance.

Новые модули и изменённые GUI модули проходят scoped Ruff. Расширенный прогон
также показал 39 F401 в совместимом facade `layer_management.py`; отдельный
`git show HEAD` baseline дал те же 39 F401. Квитанции
`profile-lint-baseline/{head,current}.json`. Legacy re-export imports не удалялись
ради линтера; полный Ruff green для этого facade не заявляется.

## Обычный UI: внешнее ограничение

Обычное audit-окно не запускалось, поскольку управление окнами недоступно.
Три проверки `@oai/sky` (повтор и reset runtime включены) завершились:
`Computer Use native pipe is unavailable ... file not found (WinError 2)`.
Резервный `cua.getState()` вернул тот же сбой native inventory, `apps=[]`.
Это не доказательство отсутствия приложений: инвентаризация не сработала.

Состояние сохранено в `ordinary-ui-audit/computer-use-unavailable.json`;
подготовленный профиль и guarded install/normal-launch/uninstall script
сохранены. `lifecycle-1.json` отсутствует, новый обычный процесс не запускался.
Логин, ordinary UI process restart и полный installed UI E2E не засчитываются
как пройденные. Auth UI не автоматизировался; обход helper через Win32/UIA
или отключение frozen login guard не вводились.

Цель остаётся активной; другие функциональные/интеграционные требования можно
продолжать независимо от доступности native UI helper.
