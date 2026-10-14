"""model_train_and_recognition: render chart image."""
from __future__ import annotations
import math
from pathlib import Path
from collections.abc import Callable
from PIL import Image, ImageDraw, ImageFont


class RenderChartImageMixin:
    @staticmethod
    def _render_chart_image(
        *,
        save_path: Path,
        title: str,
        x_label: str,
        y_label: str,
        series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]],
        y_formatter: Callable[[float], str] | None = None,
    ) -> None:
        non_empty_series = [(label, points, color) for label, points, color in series if points]
        if not non_empty_series:
            return

        width, height = 1280, 720
        margin_left, margin_right = 100, 40
        margin_top, margin_bottom = 70, 90
        plot_left = margin_left
        plot_top = margin_top
        plot_right = width - margin_right
        plot_bottom = height - margin_bottom
        plot_width = max(1, plot_right - plot_left)
        plot_height = max(1, plot_bottom - plot_top)
        background_color = (255, 255, 255)
        axis_color = (40, 40, 40)
        grid_color = (222, 228, 236)
        text_color = (25, 25, 25)

        all_points = [point for _label, points, _color in non_empty_series for point in points]
        x_values = [float(point[0]) for point in all_points]
        y_values = [float(point[1]) for point in all_points]
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        if math.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0
        if math.isclose(y_min, y_max):
            delta = 1.0 if math.isclose(y_min, 0.0) else abs(y_min) * 0.1
            y_min -= delta
            y_max += delta
        if y_min > 0.0:
            y_min = 0.0

        x_span = max(1e-9, x_max - x_min)
        y_span = max(1e-9, y_max - y_min)
        format_y = y_formatter or (lambda value: f'{float(value):.4f}')

        image = Image.new('RGB', (width, height), color=background_color)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        draw.text((plot_left, 24), str(title), fill=text_color, font=font)

        tick_count = 5
        for tick_index in range(tick_count + 1):
            x = plot_left + (plot_width * tick_index / tick_count)
            y = plot_top + (plot_height * tick_index / tick_count)
            draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
            draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)

            x_value = x_min + (x_span * tick_index / tick_count)
            y_value = y_max - (y_span * tick_index / tick_count)
            x_text = f'{x_value:.0f}' if abs(x_value - round(x_value)) < 1e-6 else f'{x_value:.2f}'
            y_text = format_y(y_value)
            draw.text((x - 12, plot_bottom + 12), x_text, fill=text_color, font=font)
            draw.text((12, y - 6), y_text, fill=text_color, font=font)

        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
        draw.text((plot_right - 50, plot_bottom + 40), str(x_label), fill=text_color, font=font)
        draw.text((16, plot_top - 24), str(y_label), fill=text_color, font=font)

        for series_index, (label, points, color) in enumerate(non_empty_series):
            scaled_points: list[tuple[float, float]] = []
            for x_value, y_value in points:
                normalized_x = (float(x_value) - x_min) / x_span
                normalized_y = (float(y_value) - y_min) / y_span
                scaled_x = plot_left + (normalized_x * plot_width)
                scaled_y = plot_bottom - (normalized_y * plot_height)
                scaled_points.append((scaled_x, scaled_y))
            if len(scaled_points) == 1:
                x_coord, y_coord = scaled_points[0]
                draw.ellipse((x_coord - 3, y_coord - 3, x_coord + 3, y_coord + 3), fill=color)
            else:
                draw.line(scaled_points, fill=color, width=3)
                for x_coord, y_coord in scaled_points:
                    draw.ellipse((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2), fill=color)

            legend_x = plot_right - 220
            legend_y = 24 + (series_index * 22)
            draw.line((legend_x, legend_y + 7, legend_x + 24, legend_y + 7), fill=color, width=3)
            draw.text((legend_x + 32, legend_y), str(label), fill=text_color, font=font)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
