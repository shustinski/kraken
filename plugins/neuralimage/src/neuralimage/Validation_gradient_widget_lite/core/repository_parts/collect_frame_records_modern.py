"""repository: collect frame records modern."""
from __future__ import annotations
from pathlib import Path
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildOptions,
    BuildResult,
    FolderSpec,
    FrameRecord,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.analytics_worker_count import (
    _build_folder_path_lookup,
    _resolve_aux_path_from_lookup,
    build_folder_index,
    natural_sort_key,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import (
    BuildCancelledError,
    build_frame_identity,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.compute_record_payloads import _available_metric_keys_for_models

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.record_payload_cache_key import _resolve_model_path_for_key

def _collect_frame_records_modern(
    model_specs: tuple[ModelSpec, ...],
    options: BuildOptions,
    *,
    original_folder: FolderSpec | None = None,
    gt_folder: FolderSpec | None = None,
    cancel_check=None,
    progress_callback=None,
) -> BuildResult:
    if not model_specs:
        raise ValueError("At least one model folder must be selected.")

    extensions = tuple(str(ext).lower() for ext in options.file_extensions)
    base_spec = model_specs[0]
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError("Build cancelled")
    if progress_callback is not None:
        progress_callback(0, 0, "Indexing base folder")
    base_index = build_folder_index(
        base_spec.mask_folder,
        recursive=options.recursive,
        extensions=extensions,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        progress_interval=max(1, int(options.progress_update_interval)),
    )
    if not base_index:
        raise ValueError("Selected model folders do not contain matching image frames.")

    if progress_callback is not None:
        progress_callback(0, len(base_index), "Indexing auxiliary folders")
    original_index = build_folder_index(
        original_folder.path,
        recursive=options.recursive,
        extensions=extensions,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        progress_interval=max(1, int(options.progress_update_interval)),
    ) if original_folder is not None else {}
    gt_index = build_folder_index(
        gt_folder.path,
        recursive=options.recursive,
        extensions=extensions,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        progress_interval=max(1, int(options.progress_update_interval)),
    ) if gt_folder is not None else {}
    original_lookup = _build_folder_path_lookup(original_index)
    gt_lookup = _build_folder_path_lookup(gt_index)
    fallback_model_indexes: dict[str, _FolderPathLookup] = {}
    fallback_prob_indexes: dict[str, _FolderPathLookup] = {}

    records: list[FrameRecord] = []
    sorted_keys = sorted(base_index.keys(), key=natural_sort_key)
    for index, key in enumerate(sorted_keys):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        model_mask_paths: dict[str, str] = {base_spec.model_id: str(base_index[key])}
        model_prob_paths: dict[str, str] = {}
        if base_spec.prob_folder is not None:
            resolved_prob = _resolve_model_path_for_key(
                base_spec.prob_folder,
                key,
                extensions,
                fallback_prob_indexes,
                recursive=options.recursive,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            model_prob_paths[base_spec.model_id] = str(resolved_prob) if resolved_prob is not None else ""
        else:
            model_prob_paths[base_spec.model_id] = ""
        missing_required_model = False
        for spec in model_specs[1:]:
            resolved = _resolve_model_path_for_key(
                spec.mask_folder,
                key,
                extensions,
                fallback_model_indexes,
                recursive=options.recursive,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            if resolved is None:
                missing_required_model = True
                break
            model_mask_paths[spec.model_id] = str(resolved)
            if spec.prob_folder is not None:
                resolved_prob = _resolve_model_path_for_key(
                    spec.prob_folder,
                    key,
                    extensions,
                    fallback_prob_indexes,
                    recursive=options.recursive,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                )
                model_prob_paths[spec.model_id] = str(resolved_prob) if resolved_prob is not None else ""
            else:
                model_prob_paths[spec.model_id] = ""
        if missing_required_model:
            continue
        original_path = _resolve_aux_path_from_lookup(key, original_lookup)
        gt_path = _resolve_aux_path_from_lookup(key, gt_lookup)
        records.append(FrameRecord(
            key=key,
            display_name=Path(key).name,
            identity=build_frame_identity(key, index),
            first_path=str(base_index[key]),
            second_path=str(model_mask_paths.get(model_specs[1].model_id, model_mask_paths[base_spec.model_id])) if len(model_specs) > 1 else str(base_index[key]),
            base_path=str(original_path) if original_path is not None else None,
            original_path=str(original_path) if original_path is not None else None,
            gt_path=str(gt_path) if gt_path is not None else None,
            model_mask_paths=model_mask_paths,
            model_prob_paths=model_prob_paths,
        ))
        if progress_callback is not None and (index == 0 or (index + 1) % max(1, int(options.progress_update_interval)) == 0 or index + 1 == len(sorted_keys)):
            progress_callback(index + 1, len(sorted_keys), key)
    return BuildResult(
        records=tuple(records),
        model_specs=model_specs,
        original_folder=original_folder,
        gt_folder=gt_folder,
        first_folder=FolderSpec(path=model_specs[0].mask_folder, label=model_specs[0].display_name) if len(model_specs) > 0 else None,
        second_folder=FolderSpec(path=model_specs[1].mask_folder, label=model_specs[1].display_name) if len(model_specs) > 1 else None,
        base_folder=original_folder,
        options=options,
        min_score=0.0,
        max_score=0.0,
        eligible_key_count=len(records),
        scores_computed=False,
        best_match_key=None,
        min_absolute_score=None,
        max_absolute_score=None,
        selected_metric_key="overall_frame_score",
        available_metric_keys=_available_metric_keys_for_models(model_specs),
    )
