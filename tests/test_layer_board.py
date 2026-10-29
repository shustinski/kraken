"""Layer-board naming, selection actions, CIF order and thumbnail legend."""

from __future__ import annotations

from kraken_manager.domain.layer_board import (
    ASSIGNMENT_ACTION,
    BoardNode,
    companion_export_plan,
    collapsed_block_title,
    matrix_legend,
    multi_selection_actions,
    selection_kind,
    next_image_representation_name,
    order_review_images,
    parameter_lines,
    review_image_ids,
    workflow_view_signature,
)


def test_collapsed_block_uses_the_created_step_name() -> None:
    nodes = (
        BoardNode("job", "job", "Подготовка выборки", capability="frames.dataset.prepare.v1"),
        BoardNode("out", "dataset", "Выборка Contour"),
    )
    assert collapsed_block_title(nodes) == "Подготовка выборки · Выборка Contour"
    assert collapsed_block_title((nodes[0],)) == "Подготовка выборки"
    assert "Чёрный ящик" not in collapsed_block_title(nodes)
    assert "Черный ящик" not in collapsed_block_title(nodes)


def test_dataset_preparation_is_a_parameter_block() -> None:
    assert parameter_lines({"epochs": 4, "source_representation_id": "hidden"}, capability="frames.dataset.prepare.v1") == (
        "epochs: 4",
    )
    assert parameter_lines({"plugin_job_id": "job"}, capability="frames.dataset.prepare.v1") == ("Параметры: не заданы",)
    assert parameter_lines({}, capability="frames.vectorize.v1") == ()


def test_multi_selection_actions_follow_the_two_combinations() -> None:
    images = (
        BoardNode("a", "source", "Изображение A"),
        BoardNode("b", "binary", "Изображение B"),
    )
    vector = BoardNode("v", "vector", "Векторы Contour")
    cif = BoardNode("c", "cif", "Вариант 1")
    another_cif = BoardNode("d", "cif", "Вариант 2")
    paired = multi_selection_actions((*images, vector))
    assert paired is not None
    assert tuple(item.label for item in paired) == ("Открыть в Contour", "Выдать в работу", "Выгрузить")
    assert multi_selection_actions((*images, cif)) == ()
    exported = multi_selection_actions((cif, another_cif))
    assert exported is not None
    assert tuple(item.code for item in exported) == ("export_selection",)
    with_images = multi_selection_actions((*images, cif, another_cif))
    assert with_images is not None
    assert tuple(item.code for item in with_images) == ("export_selection",)
    assert multi_selection_actions((vector, BoardNode("v2", "vector", "Ещё вектор"))) == ()
    assert multi_selection_actions((images[0],)) is None
    assert multi_selection_actions((*images, BoardNode("job", "job", "Шаг"))) == ()
    assert selection_kind("vector", "CIF") == "cif"
    assert selection_kind("vector", "succeeded") == "vector"


def test_cif_review_order_moves_and_keeps_later_assignments() -> None:
    order = order_review_images((), "image-a", 0)
    order = order_review_images(order, "image-b", 0)
    order = order_review_images(order, "image-a", 1)
    assert order == ("image-b", "image-a")
    order = order_review_images(order, "image-b", 5)
    assert order == ("image-a", "image-b")
    events = (
        {"payload": {"action": ASSIGNMENT_ACTION, "node_id": "cif-1", "parameters": {
            "vector_representation_ids": ["cif-1"],
            "image_representation_ids": ["image-a", "image-b"],
        }}},
        {"payload": {"action": ASSIGNMENT_ACTION, "node_id": "cif-1", "parameters": {
            "vector_representation_ids": ["cif-1", "cif-2"],
            "image_representation_ids": ["image-b", "image-a", "image-b"],
        }}},
    )
    assert review_image_ids(events, ("cif-2",)) == ("image-b", "image-a")
    assert review_image_ids(events, ("other",)) == ()
    plan = companion_export_plan(
        ("image-a", "primary", "image-b"),
        primary_image_id="primary",
        names_by_id={"image-a": ("Металл", "Вариант 1"), "image-b": ("Контакты", "Вариант 2")},
    )
    assert plan == (("Металл — Вариант 1", "image-a"), ("Контакты — Вариант 2", "image-b"))


def test_thumbnail_legend_uses_only_present_fields() -> None:
    empty = matrix_legend(())
    assert [section.title for section in empty] == ["Исполнитель", "Время", "Качество", "Статус"]
    assert all(section.empty and section.empty_text == "Нет данных" for section in empty)
    sections = {section.title: section for section in matrix_legend((
        {"performer_initials": "ИП", "performer_color": "#112233", "status": "image_ready"},
        {"performer_initials": "ИП", "performer_color": "#112233", "modified_at": "2026-01-01T00:00:00+00:00", "quality": 0.2},
        {"performer_name": "Иванов", "modified_at": "2026-02-01T00:00:00+00:00", "quality": None, "review_status": "checked"},
    ))}
    assert [entry.label for entry in sections["Исполнитель"].entries] == ["ИП", "Иванов"]
    assert [entry.label for entry in sections["Время"].entries] == ["Раньше", "Позже"]
    assert [entry.label for entry in sections["Качество"].entries] == ["Ниже", "Выше"]
    assert [entry.label for entry in sections["Статус"].entries] == ["Изображение", "Проверено"]
    missing_quality = matrix_legend(({"status": "vectorized"},))
    quality = next(section for section in missing_quality if section.title == "Качество")
    assert quality.empty


def test_image_representation_name_and_workflow_signature() -> None:
    assert next_image_representation_name(()) == "Исходные изображения"
    assert next_image_representation_name(("Исходные изображения",)) == "Исходные изображения (2)"
    first = workflow_view_signature(
        project_revision=1,
        representation_ids=("a",),
        history_count=3,
        pipeline_revision=2,
        pipeline_status="running",
        node_statuses=(("step", "running"),),
    )
    assert first == workflow_view_signature(
        project_revision=1,
        representation_ids=("a",),
        history_count=3,
        pipeline_revision=2,
        pipeline_status="running",
        node_statuses=(("step", "running"),),
    )
    changed = workflow_view_signature(
        project_revision=1,
        representation_ids=("a",),
        history_count=3,
        pipeline_revision=3,
        pipeline_status="running",
        node_statuses=(("step", "succeeded"),),
    )
    assert first != changed
