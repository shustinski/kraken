"""manager_app: show matrix context menu."""
from __future__ import annotations
from kraken_manager.domain.identity import Permission


class ShowMatrixContextMenuMixin:
    def _show_matrix_context_menu(self, context, global_position) -> None:
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self.shell)
        selected_count = len(tuple(context.selection.coordinates(maximum=10_000)))
        export_action = menu.addAction(
            "Выгрузить файлы кадра…"
            if selected_count == 1
            else f"Выгрузить файлы выбранных кадров ({selected_count})…"
        )
        review_action = None
        if (
            selected_count
            and getattr(self, "_project_id", None) is not None
            and self._has_permission(Permission.ASSIGN_WORK)
        ):
            review_action = menu.addAction("Отправить выбранное на проверку…")
        vector_properties = None
        vector_file = None
        frame_properties = None
        if selected_count == 1:
            workspace = getattr(self, "_workspace", None)
            vector_representation_id = (
                str(workspace.vector_representation_combo.currentData() or "")
                if workspace is not None
                else ""
            )
            vector_file = next(
                (
                    item
                    for item in self._active_frame_files(context.x, context.y)
                    if str(item[1].representation_id or "")
                    == vector_representation_id
                ),
                None,
            )
            if vector_file is not None:
                vector_properties = menu.addAction("Свойства вектора…")
            frame_properties = menu.addAction("Свойства кадра…")
        selected = menu.exec(global_position)
        if selected is export_action:
            self._export_selected_frame_files(context)
        elif review_action is not None and selected is review_action:
            self.send_selection_for_review()
        elif frame_properties is not None and selected is frame_properties:
            workspace = self._workspace
            if workspace is not None:
                workspace.matrix_view.set_selection(
                    type(context.selection).single(context.x, context.y)
                )
            self.show_selected_frame()
        elif vector_properties is not None and selected is vector_properties:
            assert vector_file is not None
            self._show_frame_file_properties(*vector_file)
