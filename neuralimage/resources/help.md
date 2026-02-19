Справка по параметрам NeuralImage

1. Режимы работы

- Обучение и распознавание:
  Сначала обучает модель на выборке, затем применяет ее к исходным изображениям.
  Используйте, когда нужно получить новую модель и сразу проверить результат.

- Дообучение и распознавание:
  Загружает существующую модель, продолжает обучение и запускает распознавание.
  Используйте, когда у вас уже есть .pth модель и вы хотите улучшить качество.

- Распознавание:
  Только применение готовой модели к исходным данным.
  Используйте для быстрого инференса без обучения.

- Обучение:
  Только обучение модели без этапа распознавания.
  Подходит для подготовки модели на отдельном этапе.

2. Пути и входные данные

- Путь до исходных файлов: папка с изображениями для распознавания.
- Путь для результата: папка, куда сохраняются предсказания.
- Выборка / Исходные данные: папка обучающих изображений.
- Выборка / Метки: папка с целевыми масками.
- Путь к модели: файл .pth для дообучения/распознавания.
- Число эпох обучения: количество эпох в цикле обучения.

3. Базовые параметры

- Смещение (Step): шаг нарезки патчей, обычно 64-128.
- Размер выборки (Sample X/Y): обычно 256x256. Training Strategy

- Loss function: choose bce, dice, bce_dice, iou, bce_iou.
- Dice loss weight: weight of Dice term for bce_dice (usually 0.3-0.7).
- IoU loss weight: weight of IoU term for bce_iou (usually 0.3-0.7).
- Warmup: smooth learning-rate start in first epochs.
- Warmup epochs: usually 2-5.
- Warmup start factor: usually 0.05-0.2.
- Hard mining: samples with high loss are shown more often.
- Hard mining strength: intensity of re-sampling hard examples.
- Hard mining EMA alpha: smoothing factor of sample difficulty estimate.
- Skip all-0/all-1 labels: such samples are excluded from training step.
- Early stopping: stops when validation loss does not improve.
- Patience: usually 8-20.
- Min delta: usually 0.0001-0.001.
- Restore best weights: recommended ON.
- Batch preview: monitor image/label/output during training.

7. Предобработка

- Дополнительная обработка: включает предобработку данных.
- Размер среза края: убирает шумные границы.
- Target X/Y: приводит данные к единому размеру.

8. Мониторинг производительности

- В интерфейсе доступны метрики времени батча:
  data wait - ожидание следующего батча из DataLoader.
  forward - прямой проход модели.
  backward - обратный проход и вычисление градиентов.
  optimizer - шаг оптимизатора.
  total - суммарное время шага.
- Используйте эти метрики, чтобы понять узкое место: загрузка данных или вычисления на GPU.

9. Меню «Вид»

- Панель графиков: Train/Val кривые и динамика батчей.
- Панель настроек: показать/скрыть dock-панель параметров.
- Превью пакета обучения: видимость блока image/label/output.
- Освободить память GPU: очистка кэша CUDA после тяжелых операций.

10. Рекомендуемый старт

- Режим: Обучение и распознавание
- Sample size: 256x256
- Step: 100
- Валидация: включено, 20%
- Оптимизатор: AdamW
- Learning rate: 0.0005
- Weight decay: 0.01
- Batch size: 16
- Mixed precision: bf16 (если GPU поддерживает), иначе fp16
- Loss function: bce_dice
- Dice loss weight: 0.5
- Warmup: включено, 3 эпохи, start factor 0.1
- Hard mining: optional, usually OFF for the first baseline run
- Early stopping: включено, patience 10, min delta 0.0005
- Восстановить лучшие веса: включено

11. Если что-то пошло не так

- Нехватка памяти GPU: уменьшите Batch size, затем Sample size.
- Слабое качество: включите валидацию, уменьшите learning rate, проверьте метки.
- Медленная работа:
  если высокое data wait - уменьшите нагрузку предобработки и проверьте диск.
  если высокие forward/backward - уменьшите размер модели или батч.
  попробуйте mixed precision fp16/bf16 на CUDA.