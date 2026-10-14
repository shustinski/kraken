"""pipeline_mixin: populate pipeline operations."""
from __future__ import annotations
from contour.widget_parts._imports import (
    PIPELINE_OPERATION_GROUPS,
    QListWidgetItem,
    Qt,
    QTreeWidgetItem,
    get_operation_display_name,
)


class PopulatePipelineOperationsMixin:
    def _populate_pipeline_operations(self) -> None:
        selected_operation = self._selected_available_operation_name()
        self.operation_tree.clear()
        for _group_key, labels, operations in PIPELINE_OPERATION_GROUPS:
            group_item = QTreeWidgetItem([labels[0] if self._ui_language == "ru" else labels[1]])
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for operation_name in operations:
                child_item = QTreeWidgetItem([get_operation_display_name(operation_name, self._ui_language)])
                child_item.setData(0, Qt.ItemDataRole.UserRole, operation_name)
                summary, use_case = self._operation_help_entry(operation_name)
                child_item.setToolTip(
                    0,
                    f"{summary}\n\n"
                    + (("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case),
                )
                group_item.addChild(child_item)
            group_item.setExpanded(True)
            self.operation_tree.addTopLevelItem(group_item)
        target_operation = selected_operation or self._all_operation_names()[0]
        target_item = self._find_operation_tree_item(target_operation)
        if target_item is not None:
            self.operation_tree.setCurrentItem(target_item)
            self._update_pipeline_help_preview(target_operation)

    def _populate_pipeline_list(self) -> None:
        self._ignore_pipeline_item_change = True
        self.pipeline_list.clear()
        for step in self._pipeline.steps:
            label = get_operation_display_name(step.operation, self._ui_language)
            item = QListWidgetItem(label)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setData(Qt.ItemDataRole.UserRole, self.pipeline_list.count())
            item.setData(Qt.ItemDataRole.UserRole + 1, step.operation)
            item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
            self.pipeline_list.addItem(item)
        self._ignore_pipeline_item_change = False
        if self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(0)
            self._render_pipeline_parameters(0)
        else:
            self._clear_parameters_form()

    def _clear_parameters_form(self) -> None:
        while self.parameters_form.count():
            item = self.parameters_form.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._parameter_widgets.clear()

    def _on_pipeline_step_selected(self, row: int) -> None:
        self._render_pipeline_parameters(row)
