# Ограниченная функциональная приёмка train → model port → inference

Исправленный реальный gate: **1 passed in 21.93 s**, работа child — **20.028 s**.
Данные явно synthetic: два train и одно external validation изображение 128×128,
одна эпоха, модель `S 660k`. Оценка качества не выполнялась.

```powershell
$env:KRAKEN_RUN_MODEL_FUNCTIONAL='1'
.venv\Scripts\python.exe -m pytest tests/test_pipeline_live_neural_model.py -q --basetemp=.codex_tmp_scratch/production-audit-20261026/model-functional-03
```

Повтор требует нового basetemp. По умолчанию gate пропускается. Весь child имеет
timeout 180 s, внутренний polling — 150 s; timeout завершает собственное дерево
процессов. Windows child устанавливает `SetErrorMode(3)` до Qt. Новый длительный
job или полномасштабное обучение не создаётся.

Использованы production EmbeddedProjectService, LocalAgentRuntime,
AgentPluginGateway, NeuralImage MainWindowState/managed_training и production
result importer. Первый узел `dataset.model.train.v1` получает опубликованный
managed dataset ZIP. Второй `frames.binary-segment.v2` зависит от первого и
использует `model_input.producer_id`; первичный вход остаётся исходным изображением.

Проверены:

- actual `managed_execution.cuda_available=False`, epoch cap 1, train 2 / val 1,
  MainWindowState epochs 1;
- настоящий загружаемый checkpoint, 2682091 bytes, более 100000 параметров;
- managed SHA-256 модели и PNG, model version/SHA в provenance inference;
- lineage dataset → model и image + model → mask;
- lossless PNG 128×128 со значениями только 0/255;
- импорт обоих результатов, завершение DAG, повторное открытие артефактов
  и сохранённого состояния completed.

Receipt:
`.codex_tmp_scratch/production-audit-20261026/model-functional-03/test_actual_cpu_training_to_pi0/receipt.json`.
Модель `1c7aec9a-234a-5dbf-92f2-e055f69491cb`, SHA-256
`70f09b0467e731c993f3064fcbb620a8ba529581125fb42d4593acd348cd51f2`.
Маска `e3c0da75-0ac6-5c64-b1d1-36ec865a4a3f`, SHA-256
`f0aa957523055fe28a67c6c4a3953fb8d9efb2fee7d215a9ffd569ad01aed808`.

Выявлены два расхождения между намерением CPU и фактическим runtime. Agent
отбрасывал compute environment; теперь allowlist сохраняет только четыре
дополнительных ключа: CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS, MKL_NUM_THREADS,
OPENBLAS_NUM_THREADS. Реальный subprocess regression подтверждает их передачу
и отсутствие private database/token variables. Вместе с существующей проверкой
режимов запуска: **6 passed in 1.62 s**.

Отдельный direct torch probe на текущем Windows Torch 2.12/cu132 показал:
`CUDA_VISIBLE_DEVICES=''` присутствует в environment, но CUDA остаётся доступна;
`-1` действительно даёт False. Поэтому corrected gate использует `-1` на Windows
и проверяет фактический runtime. Первые `model-functional-01/02` сохранены как
красные receipts; CPU acceptance им не присвоена. Они были короткими runs,
а не длительным обучением. Исторические CPU claims по одному empty environment
нужно сверять с фактическими receipts, включая direct subprocess вне Agent.

Эта приёмка не доказывает качество модели, геометрии, электрической связности,
полный пользовательский корпус, UI или PostgreSQL/leased execution. Ни IoU,
ни другая метрика качества не использовалась как критерий приёмки.
