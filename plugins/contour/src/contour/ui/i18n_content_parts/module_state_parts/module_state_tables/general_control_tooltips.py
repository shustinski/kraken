"""GENERAL_CONTROL_TOOLTIPS."""
from __future__ import annotations
from typing import TYPE_CHECKING
from contour.graphics_view import EditorTool

GENERAL_CONTROL_TOOLTIPS: LocalizedTextMap = {
    "input_dir": (
        "Папка с изображениями, которые появятся в списке файлов.",
        "Folder with images that will appear in the file list.",
    ),
    "cif_dir": (
        "Папка с векторной разметкой CIF/CV для наложения на изображения. Можно оставить пустой, если разметка не нужна.",
        "Folder with CIF/CV vector annotations to overlay on images. Leave empty if annotations are not needed.",
    ),
    "output_dir": (
        "Папка, куда сохраняются результаты обработки и векторизации.",
        "Folder where processing and vectorization results are saved.",
    ),
    "dataset_dir": (
        "Папка датасета, куда экспортируется текущий кадр в режиме подготовки данных.",
        "Dataset folder where the current frame is exported during dataset preparation.",
    ),
    "browse_input": (
        "Загрузить папку базового слоя.",
        "Load the base-layer image folder.",
    ),
    "browse_cif": (
        "Выбрать папку с CIF/CV-разметкой для наложения.",
        "Choose the folder with CIF/CV annotations for overlay.",
    ),
    "browse_output": (
        "Выбрать папку для сохранения результатов.",
        "Choose the folder for saved results.",
    ),
    "browse_dataset": (
        "Выбрать папку датасета для экспорта кадров.",
        "Choose the dataset folder for frame export.",
    ),
    "refresh_files": (
        "Перечитать список изображений из выбранной входной папки.",
        "Reload the image list from the selected input folder.",
    ),
    "pick_input_images": (
        "Выбрать кадры базового слоя.",
        "Pick base-layer frame files.",
    ),
    "merge_cif_files": (
        "Добавить в индекс отдельные .cif или .cv файлы (по имени сопоставляются с изображением).",
        "Add individual .cif or .cv files to the overlay index (matched by basename to images).",
    ),
    "sidebar_list_mode": (
        "Переключить список справа между кадрами и векторами из папки CIF.",
        "Switch the sidebar between image frames and CIF overlays from the overlay folder.",
    ),
    "vector_list_sidebar": (
        "Файлы CIF из текущего индекса. Выбор переходит на кадр с тем же именем.",
        "CIF entries from the current index. Choosing a row selects the matching image basename.",
    ),
    "reload_selected_cif_overlays": (
        "Сбросить кэш и перечитать CIF для строк, выделенных в этом списке.",
        "Clear caches and reload CIF overlays for the selected sidebar rows.",
    ),
    "reload_cif_for_selected_frames": (
        "Сбросить кэш и перечитать CIF для выделенных кадров (или текущего).",
        "Clear caches and reload CIF overlays for selected image frames (or the current one).",
    ),
    "frame_nav_previous": (
        "Перейти к предыдущему кадру списка.",
        "Jump to the previous frame in the list.",
    ),
    "frame_nav_next": (
        "Перейти к следующему кадру списка.",
        "Jump to the next frame in the list.",
    ),
    "frame_nav_jump": (
        "Выбрать кадр по номеру (1 … N в текущем списке изображений).",
        "Select a frame by index (1 … N within the loaded image list).",
    ),
    "image_list": (
        "Список найденных изображений. Выбор файла открывает его для просмотра и обработки.",
        "List of found images. Selecting a file opens it for viewing and processing.",
    ),
    "process_current": (
        "Обработать только выбранное изображение текущими настройками.",
        "Process only the selected image with the current settings.",
    ),
    "start_batch": (
        "Запустить обработку всех изображений из списка.",
        "Start processing all images in the list.",
    ),
    "antialias_all_vectors": (
        "Сгладить все векторные файлы CIF в проекте.",
        "Antialias every CIF vector file in the project.",
    ),
    "stop_batch": (
        "Остановить пакетную обработку после текущих выполняемых задач.",
        "Stop batch processing after the currently running tasks finish.",
    ),
    "save_current": (
        "Сохранить результат для текущего изображения в выходную папку.",
        "Save the current image result to the output folder.",
    ),
    "save_cif": (
        "Сохранять векторную разметку в CIF.",
        "Save vector annotations as CIF.",
    ),
    "save_cv": (
        "Сохранять векторную разметку в текстовый .cv файл",
        "Save vector annotations as text .cv file.",
    ),
    "export_dataset": (
        "Экспортировать текущий кадр и разметку в папку датасета.",
        "Export the current frame and annotation to the dataset folder.",
    ),
    "dataset_mode": (
        "После сохранения помечает кадр как подготовленный для датасета и помогает не перепутать уже обработанные файлы.",
        "After saving, marks the frame as prepared for the dataset and helps distinguish already processed files.",
    ),
    "max_workers": (
        "Сколько изображений можно обрабатывать параллельно в пакетном режиме. Больше потоков быстрее, но сильнее нагружает компьютер.",
        "How many images can be processed in parallel during batch mode. More workers can be faster but load the computer more.",
    ),
    "save_svg": (
        "Сохранять SVG-файл с найденными полигонами вместе с результатом.",
        "Save an SVG file with detected polygons together with the result.",
    ),
    "save_preview": (
        "Сохранять картинку предпросмотра с наложенными полигонами.",
        "Save a preview image with polygons overlaid.",
    ),
    "external_color": (
        "Цвет внешней границы обычных полигонов на предпросмотре.",
        "Color of regular polygon outer borders in the preview.",
    ),
    "hole_color": (
        "Цвет внутренних отверстий полигонов на предпросмотре.",
        "Color of polygon inner holes in the preview.",
    ),
    "selected_color": (
        "Цвет полигона, который сейчас выбран в редакторе.",
        "Color of the polygon currently selected in the editor.",
    ),
    "via_selection_color": (
        "Цвет контакта, который сейчас выбран в редакторе.",
        "Color of the contact currently selected in the editor.",
    ),
    "conductor_hover_highlight": (
        "Цвет контура проводника при наведении указателя на проводник, отверстие в нём или контакт.",
        "Outline color used for the trace when the pointer hovers the trace, a hole inside it, or a contact.",
    ),
    "vertex_color": (
        "Цвет точек-вершин, которые показываются на полигонах.",
        "Color of vertex points shown on polygons.",
    ),
    "line_width": (
        "Толщина линий полигонов на экране. Не меняет координаты и результат векторизации.",
        "Polygon line width on screen. Does not change coordinates or vectorization results.",
    ),
    "vertex_size": (
        "Размер точек-вершин на экране. Не влияет на геометрию полигонов.",
        "Size of vertex points on screen. Does not affect polygon geometry.",
    ),
    "fill_opacity": (
        "Прозрачность заливки полигонов на предпросмотре. Ноль скрывает заливку, единица делает ее непрозрачной.",
        "Polygon fill opacity in the preview. Zero hides the fill, one makes it opaque.",
    ),
    "show_vertices": (
        "Показывать точки-вершины полигонов в редакторе.",
        "Show polygon vertex points in the editor.",
    ),
    "show_labels": (
        "Показывать номера полигонов на изображении.",
        "Show polygon IDs on the image.",
    ),
    "polygon_mode": (
        "Способ создания полигона: по отдельным точкам или прямоугольником.",
        "Polygon creation method: point by point or as a rectangle.",
    ),
    "brush_mode": (
        "Режим движения кисти. Свободный рисует как ведете мышь, режим 45 градусов ограничивает направление.",
        "Brush movement mode. Freeform follows the mouse, 45-degree mode constrains direction.",
    ),
    "brush_size": (
        "Толщина кисти в пикселях изображения. Круг под курсором показывает этот размер.",
        "Brush width in image pixels. The circle under the cursor shows this size.",
    ),
    "delete_vertex_mode": (
        "Как удалять вершины: одну ближайшую точку или все точки внутри области.",
        "How vertices are deleted: one nearest point or all points inside an area.",
    ),
    "editor_via_width": (
        "Ширина контакта, который ставится инструментом «Контакт» при клике по изображению.",
        "Width of the contact placed by the Contact tool when clicking the image.",
    ),
    "editor_via_height": (
        "Высота контакта, который ставится инструментом «Контакт» при клике по изображению.",
        "Height of the contact placed by the Contact tool when clicking the image.",
    ),
}

if TYPE_CHECKING:
    from contour.ui.i18n_content_parts.module_state_parts.module_state import LocalizedTextMap
