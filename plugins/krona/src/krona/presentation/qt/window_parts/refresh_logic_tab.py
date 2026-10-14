"""window: refresh logic tab."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QTableWidgetItem,

)


class RefreshLogicTabMixin:
    def refresh_logic_tab(self) -> None:
        self._logic_highlight_targets = {}
        self.logic_highlight_combo.blockSignals(True)
        self.logic_highlight_combo.clear()
        self.logic_truth_table.setRowCount(0)
        self.logic_truth_table.setColumnCount(0)
        if not self._logic_data:
            self.logic_status_label.setText(self._t("logic.status.unavailable", "Logic function data unavailable for current file."))
            self.logic_simplified_label.setText(
                self._t("logic.label.simplified.prefix", "Simplified: ") + self._t("logic.label.none", "-")
            )
            self.logic_sop_label.setText(self._t("logic.label.sop.prefix", "SOP: ") + self._t("logic.label.none", "-"))
            self.logic_sequential_label.setText(
                self._t("logic.label.sequential.prefix", "Sequential elements: ") + self._t("logic.label.none", "-")
            )
            self.logic_highlight_combo.blockSignals(False)
            self._clear_logic_highlight()
            return

        outputs = self._logic_data.get("outputs", {})
        sequential_elements = list(self._logic_data.get("sequential_elements", []))
        if not outputs and not sequential_elements:
            self.logic_status_label.setText(
                self._t(
                    "logic.status.no_logic",
                    "No outputs or sequential elements detected for current file.",
                )
            )
            self.logic_simplified_label.setText(
                self._t("logic.label.simplified.prefix", "Simplified: ") + self._t("logic.label.none", "-")
            )
            self.logic_sop_label.setText(self._t("logic.label.sop.prefix", "SOP: ") + self._t("logic.label.none", "-"))
            self.logic_sequential_label.setText(
                self._t("logic.label.sequential.prefix", "Sequential elements: ") + self._t("logic.label.none", "-")
            )
            self.logic_highlight_combo.blockSignals(False)
            self._clear_logic_highlight()
            return

        input_count = len(self._logic_data.get("inputs", []))
        meta = self._logic_data.get("meta", {})
        self.logic_status_label.setText(
            self._t(
                "logic.status.summary.extended",
                "Inputs: {input_count}, Outputs: {output_count}, Sequential: {sequential_count}, Transistors: {transistor_count}",
                input_count=input_count,
                output_count=len(outputs),
                sequential_count=meta.get("sequential_count", len(sequential_elements)),
                transistor_count=meta.get("transistor_count", 0),
            )
        )
        input_names = self._logic_data.get("inputs", [])
        output_names = list(outputs.keys())
        for output_name in output_names:
            label = f"OUT: {output_name}"
            self.logic_highlight_combo.addItem(label)
            self._logic_highlight_targets[label] = outputs.get(output_name, {}).get("logic_path", {})
        for element in sequential_elements:
            instance_name = str(element.get("instance") or "?")
            kind_name = str(element.get("kind") or "sequential")
            subtype = str(element.get("subtype") or "").strip()
            type_text = f"{subtype}-{kind_name}" if subtype and subtype != "unknown" else kind_name
            label = f"SEQ: {instance_name} ({type_text})"
            self.logic_highlight_combo.addItem(label)
            self._logic_highlight_targets[label] = element.get("highlight") or element.get("logic_path") or {}
        self.logic_highlight_combo.blockSignals(False)
        if self.logic_highlight_combo.count():
            self.logic_highlight_combo.setCurrentIndex(0)

        if output_names:
            reference_truth_table = outputs[output_names[0]].get("truth_table", [])
            self.logic_truth_table.setColumnCount(len(input_names) + len(output_names))
            self.logic_truth_table.setHorizontalHeaderLabels([*input_names, *output_names])
            self.logic_truth_table.setRowCount(0)

            for row_idx, row_data in enumerate(reference_truth_table):
                row = self.logic_truth_table.rowCount()
                self.logic_truth_table.insertRow(row)
                for col, input_name in enumerate(input_names):
                    value = row_data.get("inputs", {}).get(input_name)
                    self.logic_truth_table.setItem(row, col, QTableWidgetItem("1" if value == 1 else "0"))
                for out_col, output_name in enumerate(output_names):
                    output_rows = outputs.get(output_name, {}).get("truth_table", [])
                    out_value = output_rows[row_idx].get("value") if row_idx < len(output_rows) else None
                    rendered = self._t("logic.table.unknown_value", "X") if out_value is None else str(out_value)
                    self.logic_truth_table.setItem(row, len(input_names) + out_col, QTableWidgetItem(rendered))

        expr_parts = []
        sop_parts = []
        for name in output_names:
            output_data = outputs.get(name, {})
            simplified = output_data.get("simplified_expression")
            sop = output_data.get("sum_of_products")
            if simplified:
                expr_parts.append(f"{name}: {simplified}")
            if sop:
                sop_parts.append(f"{name}: {sop}")
        self.logic_simplified_label.setText(
            self._t("logic.label.simplified.prefix", "Simplified: ")
            + (" | ".join(expr_parts) if expr_parts else self._t("logic.label.none", "-"))
        )
        self.logic_sop_label.setText(
            self._t("logic.label.sop.prefix", "SOP: ")
            + (" | ".join(sop_parts) if sop_parts else self._t("logic.label.none", "-"))
        )
        self.logic_sequential_label.setText(
            self._t("logic.label.sequential.prefix", "Sequential elements: ")
            + self._render_sequential_elements_summary(sequential_elements)
        )
        if self.logic_highlight_combo.currentText():
            self._on_logic_highlight_output_changed(self.logic_highlight_combo.currentText())
        else:
            self._clear_logic_highlight()
