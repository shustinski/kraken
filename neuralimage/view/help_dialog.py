from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lib.runtime_paths import resolve_resource_path
from lib.ui_texts import get_ui_section
from lib.version import APP_VERSION


@dataclass(slots=True)
class HelpEntry:
    identifier: str
    title: str
    content: str


@dataclass(slots=True)
class HelpSection:
    identifier: str
    title: str
    intro: str = ''
    entries: list[HelpEntry] = field(default_factory=list)


class AutoSizingTextBrowser(QTextBrowser):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().documentLayout().documentSizeChanged.connect(self._sync_height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_height()

    def _sync_height(self, *_args) -> None:
        self.document().setTextWidth(max(0, self.viewport().width()))
        document_height = math.ceil(self.document().size().height())
        self.setFixedHeight(max(0, document_height + (self.frameWidth() * 2) + 4))


class HelpAccordion(QWidget):
    def __init__(self, title: str, content: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._toggle_button = QToolButton(self)
        self._toggle_button.setText(title)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_button.clicked.connect(self.set_expanded)

        self._content_view = AutoSizingTextBrowser(self)
        self._content_view.setReadOnly(True)
        self._content_view.setOpenExternalLinks(True)
        self._content_view.setFrameShape(QFrame.Shape.NoFrame)
        self._content_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content_view.setMarkdown(content.strip())
        self._content_view.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._content_view)

    def set_expanded(self, expanded: bool) -> None:
        self._toggle_button.setChecked(bool(expanded))
        self._toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content_view.setVisible(bool(expanded))

    def is_expanded(self) -> bool:
        return bool(self._toggle_button.isChecked())


class HelpDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._texts = get_ui_section('help_dialog')
        self._catalog_targets: dict[str, QWidget] = {}
        self._section_accordions: dict[str, list[HelpAccordion]] = {}
        self.setModal(False)

        self.setWindowTitle(str(self._texts.get('window_title', 'Справка')))
        self.resize(1180, 760)

        title, intro, sections = self._load_help_document(str(self._texts.get('content', '')))

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        title_label = QLabel(title or str(self._texts.get('header_title', 'Справка по проекту')), self)
        title_label.setStyleSheet('font-size: 20px; font-weight: 700;')
        header_row.addWidget(title_label, 1)

        version_template = str(self._texts.get('version_template', 'Версия: {version}'))
        version_label = QLabel(version_template.format(version=APP_VERSION), self)
        version_label.setStyleSheet('color: #7f8c8d;')
        header_row.addWidget(version_label, 0, Qt.AlignmentFlag.AlignRight)
        root_layout.addLayout(header_row)

        tools_row = QHBoxLayout()
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.setSpacing(8)
        catalog_label = QLabel(str(self._texts.get('catalog_title', 'Каталог разделов')), self)
        catalog_label.setStyleSheet('font-weight: 600;')
        tools_row.addWidget(catalog_label)
        tools_row.addStretch(1)

        expand_all_button = QPushButton(str(self._texts.get('expand_all', 'Развернуть все')), self)
        expand_all_button.clicked.connect(self._expand_all_sections)
        tools_row.addWidget(expand_all_button)

        collapse_all_button = QPushButton(str(self._texts.get('collapse_all', 'Свернуть все')), self)
        collapse_all_button.clicked.connect(self._collapse_all_sections)
        tools_row.addWidget(collapse_all_button)
        root_layout.addLayout(tools_row)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        self.catalog_tree = QTreeWidget(self)
        self.catalog_tree.setHeaderHidden(True)
        self.catalog_tree.setIndentation(16)
        self.catalog_tree.itemClicked.connect(self._handle_catalog_click)
        splitter.addWidget(self.catalog_tree)

        self.content_scroll = QScrollArea(self)
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        splitter.addWidget(self.content_scroll)
        splitter.setSizes([260, 860])

        content_container = QWidget(self)
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(14)
        self.content_scroll.setWidget(content_container)

        intro_view = self._create_markdown_view(intro)
        content_layout.addWidget(intro_view)
        self._catalog_targets['__intro__'] = intro_view
        overview_item = QTreeWidgetItem([str(self._texts.get('overview_title', 'Обзор'))])
        overview_item.setData(0, Qt.ItemDataRole.UserRole, '__intro__')
        self.catalog_tree.addTopLevelItem(overview_item)

        for section_index, section in enumerate(sections, start=1):
            section_widget = QWidget(self)
            section_layout = QVBoxLayout(section_widget)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(8)

            section_title = QLabel(section.title, self)
            section_title.setStyleSheet('font-size: 18px; font-weight: 700;')
            section_layout.addWidget(section_title)
            self._catalog_targets[section.identifier] = section_widget

            if section.intro.strip():
                section_layout.addWidget(self._create_markdown_view(section.intro))

            section_item = QTreeWidgetItem([section.title])
            section_item.setData(0, Qt.ItemDataRole.UserRole, section.identifier)
            self.catalog_tree.addTopLevelItem(section_item)

            accordions: list[HelpAccordion] = []
            for entry_index, entry in enumerate(section.entries, start=1):
                accordion = HelpAccordion(entry.title, entry.content, self)
                section_layout.addWidget(accordion)
                accordions.append(accordion)
                self._catalog_targets[entry.identifier] = accordion

                entry_item = QTreeWidgetItem([entry.title])
                entry_item.setData(0, Qt.ItemDataRole.UserRole, entry.identifier)
                section_item.addChild(entry_item)

                if section_index == 1 and entry_index == 1:
                    accordion.set_expanded(True)

            self._section_accordions[section.identifier] = accordions
            section_item.setExpanded(True)
            content_layout.addWidget(section_widget)

        content_layout.addStretch(1)
        self.catalog_tree.expandAll()
        self.catalog_tree.setCurrentItem(overview_item)

        close_button = QPushButton(str(self._texts.get('close_button', 'Закрыть')), self)
        close_button.clicked.connect(self.accept)
        root_layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

    def _handle_catalog_click(self, item: QTreeWidgetItem, _column: int) -> None:
        target_id = str(item.data(0, Qt.ItemDataRole.UserRole) or '').strip()
        if not target_id:
            return
        target_widget = self._catalog_targets.get(target_id)
        if target_widget is None:
            return
        if isinstance(target_widget, HelpAccordion):
            target_widget.set_expanded(True)
        self.content_scroll.ensureWidgetVisible(target_widget, 0, 48)

    def _expand_all_sections(self) -> None:
        for accordions in self._section_accordions.values():
            for accordion in accordions:
                accordion.set_expanded(True)

    def _collapse_all_sections(self) -> None:
        for accordions in self._section_accordions.values():
            for accordion in accordions:
                accordion.set_expanded(False)

    @staticmethod
    def _create_markdown_view(content: str) -> AutoSizingTextBrowser:
        view = AutoSizingTextBrowser()
        view.setReadOnly(True)
        view.setOpenExternalLinks(True)
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setMarkdown(content.strip())
        return view

    def _load_help_document(self, content_ref: str) -> tuple[str, str, list[HelpSection]]:
        markdown_path = self._resolve_markdown_path(content_ref)
        if markdown_path is None or not markdown_path.exists():
            return '', '', []
        return self._parse_help_markdown(markdown_path.read_text(encoding='utf-8'))

    @staticmethod
    def _resolve_markdown_path(content_ref: str) -> Path | None:
        value = str(content_ref or '').strip()
        if not value:
            return None
        md_path = Path(value)
        if not md_path.is_absolute():
            if md_path.parts and md_path.parts[0] == 'resources':
                md_path = resolve_resource_path(*md_path.parts[1:])
            else:
                md_path = Path(__file__).resolve().parent.parent / md_path
        return md_path

    @staticmethod
    def _parse_help_markdown(markdown: str) -> tuple[str, str, list[HelpSection]]:
        title = ''
        intro_lines: list[str] = []
        sections: list[HelpSection] = []
        current_section: HelpSection | None = None
        current_entry: HelpEntry | None = None
        current_entry_lines: list[str] = []

        def flush_entry() -> None:
            nonlocal current_entry, current_entry_lines, current_section
            if current_entry is None or current_section is None:
                current_entry_lines = []
                return
            current_entry.content = '\n'.join(current_entry_lines).strip()
            current_section.entries.append(current_entry)
            current_entry = None
            current_entry_lines = []

        def flush_section() -> None:
            nonlocal current_section
            flush_entry()
            if current_section is not None:
                current_section.intro = current_section.intro.strip()
                sections.append(current_section)
                current_section = None

        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()
            if line.startswith('# '):
                if not title:
                    title = line[2:].strip()
                else:
                    intro_lines.append(line)
                continue

            if line.startswith('## '):
                flush_section()
                section_index = len(sections) + 1
                current_section = HelpSection(
                    identifier=f'section-{section_index}',
                    title=line[3:].strip(),
                )
                continue

            if line.startswith('### '):
                flush_entry()
                if current_section is None:
                    current_section = HelpSection(identifier='section-1', title='General')
                entry_index = len(current_section.entries) + 1
                current_entry = HelpEntry(
                    identifier=f'{current_section.identifier}-entry-{entry_index}',
                    title=line[4:].strip(),
                    content='',
                )
                continue

            if current_entry is not None:
                current_entry_lines.append(line)
            elif current_section is not None:
                current_section.intro += f'{line}\n'
            else:
                intro_lines.append(line)

        flush_section()
        intro = '\n'.join(intro_lines).strip()
        return title, intro, sections

_help_dialog_instance: HelpDialog | None = None


def _clear_help_dialog_instance(*_args) -> None:
    global _help_dialog_instance
    _help_dialog_instance = None


def show_help_dialog(parent: QWidget | None = None) -> HelpDialog:
    global _help_dialog_instance
    if _help_dialog_instance is None:
        dialog = HelpDialog(parent)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(_clear_help_dialog_instance)
        _help_dialog_instance = dialog
    else:
        dialog = _help_dialog_instance

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
