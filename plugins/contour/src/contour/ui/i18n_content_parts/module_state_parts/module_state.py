"""module_state: module state."""
from __future__ import annotations
from contour.graphics_view import EditorTool

from .module_state_tables.extraction_help_texts import EXTRACTION_HELP_TEXTS
from .module_state_tables.pipeline_parameter_help_texts import PIPELINE_PARAMETER_HELP_TEXTS
from .module_state_tables.editor_tool_tooltips import EDITOR_TOOL_TOOLTIPS
from .module_state_tables.general_control_tooltips import GENERAL_CONTROL_TOOLTIPS
from .module_state_tables.pipeline_operation_groups import PIPELINE_OPERATION_GROUPS
from .module_state_tables.pipeline_operation_help_texts import PIPELINE_OPERATION_HELP_TEXTS

LocalizedTextMap = dict[str, tuple[str, str]]



PIPELINE_CONTROL_TOOLTIPS: LocalizedTextMap = {
    "add_step_button": (
        "Добавляет выбранный фильтр из списка в текущий pipeline.",
        "Adds the selected filter from the list to the current pipeline.",
    ),
    "remove_step_button": (
        "Удаляет выбранный шаг из pipeline. Остальные шаги сохраняются.",
        "Removes the selected step from the pipeline. Other steps stay unchanged.",
    ),
    "move_up_button": (
        "Перемещает выбранный фильтр выше. Порядок важен: верхние фильтры применяются раньше.",
        "Moves the selected filter up. Order matters: upper filters run earlier.",
    ),
    "move_down_button": (
        "Перемещает выбранный фильтр ниже. Нижние фильтры применяются позже.",
        "Moves the selected filter down. Lower filters run later.",
    ),
    "auto_apply_checkbox": (
        "Автоматически пересчитывает текущее изображение после изменения pipeline или его параметров.",
        "Automatically reprocesses the current image after pipeline or parameter changes.",
    ),
    "apply_current_button": (
        "Применяет текущий pipeline к открытому изображению вручную.",
        "Manually applies the current pipeline to the open image.",
    ),
    "save_json_button": (
        "Сохраняет текущий набор фильтров и их параметры в JSON-файл.",
        "Saves the current filter list and parameters to a JSON file.",
    ),
    "load_json_button": (
        "Загружает pipeline из JSON-файла и заменяет текущий список фильтров.",
        "Loads a pipeline from a JSON file and replaces the current filter list.",
    ),
}


EDITOR_ACTION_TOOLTIPS: LocalizedTextMap = {
    "undo_button": (
        "Отменяет последнее изменение полигонов.",
        "Undoes the last polygon edit.",
    ),
    "redo_button": (
        "Возвращает последнее отмененное изменение полигонов.",
        "Redoes the last undone polygon edit.",
    ),
    "zoom_in_button": (
        "Увеличивает изображение в окне просмотра.",
        "Zooms in on the image view.",
    ),
    "zoom_out_button": (
        "Уменьшает изображение в окне просмотра.",
        "Zooms out of the image view.",
    ),
    "fit_button": (
        "Подгоняет изображение целиком под размер окна просмотра.",
        "Fits the whole image into the view.",
    ),
}



