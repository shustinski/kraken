# Changelog

- Inference pipeline v2: unified tile geometry, raised-cosine overlap blending, lossless PNG output, full-frame threshold calibration, and explicit confidence/uncertainty semantics.
- CIF-derived `binary_cif` masks are now stored atomically as true 1-bit PNG files; matching legacy JPEG cache files are removed after conversion.

## v1.0.0

- Добавлены расширенные функции потерь: `bce`, `dice`, `bce_dice`, `iou`, `bce_iou`.
- Добавлены параметры `dice/iou weight` и их управление в UI.
- Добавлен `hard mining` для более частой подачи сложных примеров.
- Добавлен пропуск сэмплов с полностью пустой/полностью заполненной маской.
- Добавлен график `train loss vs batch` со стратегией прореживания при большом количестве точек.
- Улучшена справка и добавлена информация о версии программы.

## Следующие версии

- Дополняйте этот список по мере релизов.
