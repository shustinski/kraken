"""presenter: repeated percentile entries."""
from __future__ import annotations
from bisect import bisect_right
from PyQt6.QtCore import QTimer
from ...core.domain import FrameRecord
from ...core.repository import (
    compute_metric_percentiles,
)
from ...ui.ui_constants import (
    PERCENTILE_BAND_BOUNDS,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class RepeatedPercentileEntriesMixin:
    def _component_value_text(self, value: str) -> str:
        values_en = {
            'labeled frame': 'labeled frame',
            'unlabeled frame': 'unlabeled frame',
            'supervised error map': 'supervised error map',
        }
        values_ru = {
            'labeled frame': 'размеченный кадр',
            'unlabeled frame': 'неразмеченный кадр',
            'supervised error map': 'supervised-карта ошибки',
            'variance/entropy risk map': 'карта риска по variance / entropy',
            'mean entropy of consensus probability': 'средняя энтропия consensus probability',
            'mean variance over model probability maps': 'средняя дисперсия probability maps моделей',
            'acquisition_score': 'приоритет на разметку',
        }
        values = values_ru if getattr(self._i18n, 'language', 'en') == 'ru' else values_en
        return values.get(value, value.replace(' vs ', ' против ') if getattr(self._i18n, 'language', 'en') == 'ru' else value)

    def _decorate_metric_lines(self, metric_key: str, summary, lines: list[str]) -> list[str]:
        decorated: list[str] = []
        hint = self._metric_hint(metric_key, summary)
        if hint:
            decorated.append(hint)
        for line in lines:
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            status_prefix = ''
            for status in ('active', 'auxiliary', 'legacy'):
                prefix = f'{status} '
                if stripped.startswith(prefix):
                    status_prefix = f"{self._t(f'status.{status}')} "
                    stripped = stripped[len(prefix):]
                    break
            if ':' in stripped:
                name, value = stripped.split(':', 1)
                name = self._component_name_label(name.strip())
                value = self._component_value_text(value.strip())
                stripped = f"{status_prefix}{name}: {value}"
            else:
                stripped = status_prefix + self._component_value_text(stripped)
            decorated.append(indent + stripped)
        return decorated

    def _percentile_style(self, percentile: float | None, metric_key: str | None = None) -> str:
        if percentile is None:
            return self._metric_score_style(None, "overall_polygon_score")
        clipped = max(0.0, min(float(percentile), 100.0))
        # Percentiles shown in the UI are always goodness percentiles:
        # low percentile means a worse frame, high percentile means a better one.
        if clipped < 15.0:
            background = "#8c2f39"
            foreground = "#ffe9ec"
        elif clipped < 35.0:
            background = "#a75d12"
            foreground = "#fff0dc"
        elif clipped < 60.0:
            background = "#6f7a18"
            foreground = "#f7ffd8"
        else:
            background = "#1f5f3b"
            foreground = "#e9fff1"
        return f"padding: 6px 10px; border-radius: 8px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _percentile_text(self, percentile: float | None) -> str:
        if percentile is None:
            return "-"
        return f"P{float(percentile):.1f}"

    def _percentile_map_for_metric(self, state: ExtendMatrixTabState, metric_key: str) -> dict[str, float]:
        base_records = self._base_records_for_state(state)
        cache_key = (str(metric_key), id(base_records))
        cached = state.percentile_cache.get(cache_key)
        if cached is not None:
            return cached
        percentile_map = compute_metric_percentiles(base_records, metric_key)
        state.percentile_cache[cache_key] = percentile_map
        return percentile_map

    def _percentile_histogram_counts(self, state: ExtendMatrixTabState, metric_key: str) -> list[int]:
        percentiles = self._percentile_map_for_metric(state, metric_key)
        counts = [0] * len(PERCENTILE_BAND_BOUNDS)
        upper_bounds = [float(high) for _low, high in PERCENTILE_BAND_BOUNDS[:-1]]
        last_index = len(counts) - 1
        for value in percentiles.values():
            clipped = max(0.0, min(float(value), 100.0))
            counts[min(last_index, bisect_right(upper_bounds, clipped))] += 1
        return counts

    def _repeated_percentile_entries(self, state: ExtendMatrixTabState, *, band: str) -> list[tuple[FrameRecord, int, float, list[str]]]:
        available_keys = list(self._percentile_basis_keys_for_state(state, state.build_result))
        cache_key = (str(band), tuple(str(key) for key in available_keys), id(state.build_result.records))
        cached = state.repeated_percentile_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        metrics_by_record: dict[str, list[tuple[str, float]]] = {record.key: [] for record in state.build_result.records}
        for metric_key in available_keys:
            percentile_map = self._percentile_map_for_metric(state, metric_key)
            for record in state.build_result.records:
                percentile = percentile_map.get(record.key)
                if percentile is None:
                    continue
                if band == 'bad' and float(percentile) < 15.0:
                    metrics_by_record[record.key].append((self._metric_label(metric_key, state.build_result), float(percentile)))
                elif band == 'good' and float(percentile) >= 60.0:
                    metrics_by_record[record.key].append((self._metric_label(metric_key, state.build_result), float(percentile)))
        entries: list[tuple[FrameRecord, int, float, list[str]]] = []
        records_by_key = {record.key: record for record in state.build_result.records}
        for key, values in metrics_by_record.items():
            if not values:
                continue
            metric_labels = [label for label, _percentile in values]
            average_percentile = sum(percentile for _label, percentile in values) / float(len(values))
            record = records_by_key[key]
            entries.append((record, len(values), average_percentile, metric_labels))
        if band == 'bad':
            entries.sort(key=lambda item: (-item[1], item[2], item[0].display_name.lower()))
        else:
            entries.sort(key=lambda item: (-item[1], -item[2], item[0].display_name.lower()))
        state.repeated_percentile_cache[cache_key] = tuple(entries)
        return entries

    def _update_repeated_percentile_lists(self, state: ExtendMatrixTabState) -> None:
        def summary(entries: list[tuple[FrameRecord, int, float, list[str]]]) -> tuple[int, float, int]:
            visible_entries = entries[:25]
            if not visible_entries:
                return 0, 0.0, 0
            frame_count = len(visible_entries)
            mean_hits = sum(count for _record, count, _avg_percentile, _labels in visible_entries) / float(frame_count)
            max_hits = max(count for _record, count, _avg_percentile, _labels in visible_entries)
            return frame_count, mean_hits, max_hits

        bad_entries = self._repeated_percentile_entries(state, band='bad')
        good_entries = self._repeated_percentile_entries(state, band='good')
        if state.repeated_bad_column is not None and hasattr(state.repeated_bad_column, 'set_payload'):
            frame_count, mean_hits, max_hits = summary(bad_entries)
            state.repeated_bad_column.set_payload(frame_count, mean_hits, max_hits, active=state.correlation_filter_band == 'bad')
        if state.repeated_good_column is not None and hasattr(state.repeated_good_column, 'set_payload'):
            frame_count, mean_hits, max_hits = summary(good_entries)
            state.repeated_good_column.set_payload(frame_count, mean_hits, max_hits, active=state.correlation_filter_band == 'good')

    def _update_metric_histograms(self, state: ExtendMatrixTabState) -> None:
        preview = state.preview
        if preview is None:
            return
        base_record_count = len(self._base_records_for_state(state))
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        for metric_key, card in preview.histogram_cards.items():
            visible = metric_key in available_keys
            if not visible:
                card.setVisible(False)
                continue
            counts = self._percentile_histogram_counts(state, metric_key)
            active_bin = state.percentile_filter_bin_index if state.percentile_filter_metric_key == metric_key else None
            tooltip = self._metric_hint_fallback(metric_key, state.build_result)
            card.set_payload(self._metric_label(metric_key, state.build_result), counts, base_record_count, visible=True, active_bin=active_bin, tooltip=tooltip)
        self._update_repeated_percentile_lists(state)

    def _schedule_metric_histogram_update(self, state: ExtendMatrixTabState) -> None:
        self._histogram_update_generation += 1
        generation = int(self._histogram_update_generation)
        QTimer.singleShot(0, lambda s=state, g=generation: self._update_metric_histograms_chunked(s, g, 0))

    def _update_metric_histograms_chunked(self, state: ExtendMatrixTabState, generation: int, index: int) -> None:
        if generation != self._histogram_update_generation or state.widget not in self._tab_states:
            return
        preview = state.preview
        if preview is None:
            return
        base_record_count = len(self._base_records_for_state(state))
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        histogram_items = list(preview.histogram_cards.items())
        if index >= len(histogram_items):
            QTimer.singleShot(0, lambda s=state, g=generation: self._update_repeated_percentile_lists(s) if g == self._histogram_update_generation and s.widget in self._tab_states else None)
            return
        metric_key, card = histogram_items[index]
        if metric_key not in available_keys:
            card.setVisible(False)
        else:
            counts = self._percentile_histogram_counts(state, metric_key)
            active_bin = state.percentile_filter_bin_index if state.percentile_filter_metric_key == metric_key else None
            tooltip = self._metric_hint_fallback(metric_key, state.build_result)
            card.set_payload(self._metric_label(metric_key, state.build_result), counts, base_record_count, visible=True, active_bin=active_bin, tooltip=tooltip)
        QTimer.singleShot(0, lambda s=state, g=generation, i=index + 1: self._update_metric_histograms_chunked(s, g, i))
