"""repository: collect frame records."""
from __future__ import annotations
from dataclasses import replace
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildOptions,
    BuildResult,
    FolderSpec,
    FrameRecord,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.collect_frame_records_modern import _collect_frame_records_modern

def collect_frame_records(
    model_specs: tuple[ModelSpec, ...] | FolderSpec,
    options: BuildOptions | FolderSpec,
    maybe_options: BuildOptions | None = None,
    *,
    original_folder: FolderSpec | None = None,
    gt_folder: FolderSpec | None = None,
    base_folder: FolderSpec | None = None,
    cancel_check=None,
    progress_callback=None,
) -> BuildResult:
    """Collect matched frame records in modern or legacy-lite mode."""

    if isinstance(model_specs, FolderSpec):
        first_folder = model_specs
        second_folder = options if isinstance(options, FolderSpec) else None
        build_options = maybe_options if isinstance(maybe_options, BuildOptions) else BuildOptions()
        if second_folder is None:
            raise ValueError("Legacy collect_frame_records requires two folder specs.")
        compat_model_specs = (
            ModelSpec(model_id="first", display_name=first_folder.label, mask_folder=first_folder.path, threshold=float(build_options.mask_threshold)),
            ModelSpec(model_id="second", display_name=second_folder.label, mask_folder=second_folder.path, threshold=float(build_options.mask_threshold)),
        )
        modern = _collect_frame_records_modern(
            compat_model_specs,
            build_options,
            original_folder=base_folder,
            gt_folder=None,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        records: list[FrameRecord] = []
        for record in modern.records:
            first_path = str(record.model_mask_paths.get("first", ""))
            second_path = str(record.model_mask_paths.get("second", ""))
            records.append(
                replace(
                    record,
                    first_path=first_path,
                    second_path=second_path,
                    base_path=record.original_path,
                )
            )
        return replace(
            modern,
            records=tuple(records),
            first_folder=first_folder,
            second_folder=second_folder,
            base_folder=base_folder,
            original_folder=base_folder,
        )

    resolved_options = options if isinstance(options, BuildOptions) else maybe_options
    if not isinstance(resolved_options, BuildOptions):
        raise ValueError("collect_frame_records requires BuildOptions for modern mode.")
    return _collect_frame_records_modern(
        tuple(model_specs),
        resolved_options,
        original_folder=original_folder,
        gt_folder=gt_folder,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
