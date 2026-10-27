# Удалённая карта расхождений: функциональная проверка

Нового обучения и оценки качества векторизации нет. Используются контролируемые
бинарные fixtures и действительный рендерер Karakal, а не картинка-заглушка.

Реализованы `POST/GET .../analysis-runs/{run_id}/frames/{frame_id}/map`.
POST требует RUN_PLUGIN, CAS и idempotency key, GET — членства в проекте.
Запрос создаёт отдельное задание в существующей Agent queue; канонические
`analysis_map.*` события сохраняются в том же `analysis-run:<uuid>` stream.
Отдельной БД/очереди/графа нет. Проекция исходных frame metrics не изменяется.

Берутся точные версии/SHA и recipe/parameters выбранного кадра исходного
анализа, даже если активные версии представления уже сменились. Рендер выполняет
настоящий Karakal Worker. PNG передаётся через существующий BlobStore. Agent
проверяет requested frame, путь, SHA и размер; сервер повторно проверяет SHA,
размер, PNG structure и ограничение 64 Mi pixels / 256 MiB encoded bytes.
Hub сверяет полученные байты с закреплённым SHA до отображения.

В существующей Qt панели кнопка открывает немодальную карту с прокруткой,
идентификаторами run/frame, выражением и легендой левого/правого операндов.
Это важно для выражений вроде compare(xor(A,B),C): красный не объявляется
разницей исключительно исходного A. Сетевые операции выполняются в фоне.

Проверенные сценарии:

- native Karakal → точные белый/красный/синий/чёрный пиксели → Agent digest guard;
- реальный HTTP/PG/Agent/subprocess → PNG import → GET → reopen;
- смена активного входа на заведомо непригодный PNG не меняет закреплённые входы;
- сбой submit после durable intent commit → queued intent → background repair
  даже у завершённого анализа → повтор с исходным idempotency key;
- отмена снимает queued map job; исходные metrics сохранены;
- повреждение blob отвергается HTTP 409; повреждение передачи не открывает
  ошибочную картинку в Qt;
- anonymous/outsider отказ, viewer read-only, CAS/project scope guards;
- неверный путь/SHA/размер, не-PNG, некорректный image и чрезмерные dimensions
  отвергаются; native map extension не меняет старый result payload без карты.

Evidence: `.codex_tmp_scratch/production-audit-20261026/analysis-map-recovery/`
и `analysis-map-final/` (8 модулей, 91 tests passed, 0 skipped, отдельные
logs/XML/JSON). После улучшения очереди Qt повторены ещё 5 UI tests:
`analysis-map-ui-queue-final/`. Наборы пересекаются и не суммируются.
`analysis-map-live-qt/`: 3 passed; успешный сценарий дополнительно проходит
единым путём native worker → Agent → PostgreSQL → HTTP client → настоящий
немодальный Qt dialog с проверкой пикселей, без подмены изображения в GUI.
Первые красные прогоны
сохранены: старое ожидание disabled map button и старое сравнение run revision
после добавления map events были заменены проверками нового поведения.

Локальный настоящий рендер позднее реализован и проверен:
[протокол локальной карты](2026-10-26-local-analysis-map-functional.md).
Actual worker runtime attestation, полный 20-шаговый E2E и нагрузочная
приёмка также открыты. Ранее проверенные `preserved-installers` предшествуют
этой функции; новую карту они не содержат. Финальная упаковка нужна после
завершения оставшихся изменений.
