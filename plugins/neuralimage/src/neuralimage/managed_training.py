"""Managed noninteractive training using the existing NeuralImage trainer."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import random


# Explicitly supported UI settings. Paths and external/custom generators are
# deliberately not accepted as untrusted manifest settings; validation roots are
# resolved solely inside the staged, hash-verified dataset below.
HEADLESS_SETTINGS = frozenset(
    """
step vertical_rotation horizontal_rotation flip_x flip_y additional_augmentation
augmentation_multiplier augmentation_brightness_strength augmentation_contrast_strength
augmentation_gamma_strength augmentation_noise_probability augmentation_noise_sigma
augmentation_blur_probability augmentation_blur_radius sample_size train_patch_size
model color_mode shuffle shuffle_patches_in_frame random_crop crops_per_image
scale_augmentation scale_augmentation_strength use_validation validation_percent
validation_source save_validation_binary_images batch_size dataloader_num_workers
train_batch_size sync_patch_sizes patch_batch_sync_mode overlap crop_enabled
resize_enabled edge_cut_size target_size compression_factor recursive_file_search
optimizer_name mixed_precision loss_function loss_term_weights dice_loss_weight
iou_loss_weight learning_rate weight_decay early_stopping_enabled warmup_enabled warmup_epochs
warmup_start_factor scheduler_name scheduler_plateau_factor scheduler_plateau_patience
scheduler_plateau_threshold scheduler_plateau_min_lr scheduler_plateau_cooldown
scheduler_cosine_t_max scheduler_cosine_eta_min scheduler_one_cycle_max_lr
scheduler_one_cycle_pct_start scheduler_one_cycle_anneal_strategy scheduler_one_cycle_div_factor
scheduler_one_cycle_final_div_factor scheduler_one_cycle_three_phase scheduler_step_lr_step_size
scheduler_step_lr_gamma hard_mining_enabled hard_mining_strength hard_mining_ema_alpha
random_patch_size_enabled random_patch_min_size random_patch_max_size hard_pixel_mining_enabled
hard_pixel_mining_ratio cutout_enabled cutout_probability cutout_holes cutout_size_ratio
random_artifacts_enabled random_artifacts_probability random_artifacts_count random_artifacts_size_ratio
random_artifacts_dust_enabled random_artifacts_resist_residue_enabled random_artifacts_etch_residue_enabled
random_artifacts_particle_cluster_enabled random_artifacts_flake_enabled mixup_enabled mixup_probability
mixup_alpha skip_uniform_labels rare_patch_oversampling_enabled rare_patch_oversampling_factor
use_multi_gpu multi_gpu_mode torch_compile_enabled show_batch_preview log_update_frequency
local_crop_size context_crop_size context_input_size context_branch_channels fusion_type
use_context_branch deep_supervision
""".split()
)


def validate_external_partition(images, validation_images, *, recursive=False):
    from kraken_core.safe_files import ensure_regular_directory, is_link_or_reparse, open_regular_read
    from neuralimage.kraken_bridge import KrakenBridgeError

    manifests = []
    for folder in (images, validation_images):
        folder = ensure_regular_directory(folder)
        entries = []
        for directory, directories, files in os.walk(folder, followlinks=False):
            if any(is_link_or_reparse(Path(directory) / name) for name in directories):
                raise KrakenBridgeError("Dataset partitions must not contain directory links/reparse points")
            if not recursive:
                directories[:] = []
            for name in files:
                path = Path(directory) / name
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                    continue
                digest = hashlib.sha256()
                with open_regular_read(path, root=folder) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                entries.append((path.relative_to(folder).as_posix(), digest.hexdigest()))
        if not entries:
            raise KrakenBridgeError("Training and external validation both require actual images")
        manifests.append(sorted(entries))
    if {digest for _, digest in manifests[0]} & {digest for _, digest in manifests[1]}:
        raise KrakenBridgeError(
            "Training/validation image bytes overlap; rename does not create independent validation"
        )
    return dict(
        train_images=len(manifests[0]),
        validation_images=len(manifests[1]),
        partition_image_manifest_sha256=hashlib.sha256(
            json.dumps(manifests, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def training_configuration(parameters):
    from neuralimage.kraken_bridge import KrakenBridgeError

    options = parameters.get("training", {})
    if not isinstance(options, Mapping) or set(options) - {"epochs", "seed", "settings", "validation_relative_root"}:
        raise KrakenBridgeError("training must contain only epochs, seed, settings and validation_relative_root")
    try:
        json.dumps(dict(options), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise KrakenBridgeError("training settings require finite JSON values") from exc
    epochs, seed = options.get("epochs", 20), options.get("seed", 0)
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise KrakenBridgeError("training.epochs must be an integer in1..10000")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise KrakenBridgeError("training.seed must be an unsigned32bit integer")
    settings = options.get("settings", {})
    if not isinstance(settings, Mapping) or set(settings) - HEADLESS_SETTINGS:
        raise KrakenBridgeError("Unsupported headless training setting (paths/custom generators are not accepted)")
    validation_root = options.get("validation_relative_root")
    if validation_root is not None and (not isinstance(validation_root, str) or not validation_root.strip()):
        raise KrakenBridgeError("validation_relative_root must be a nonempty staged relative path")
    return epochs, seed, dict(settings), validation_root


def run_training(session):
    from kraken_core.plugin_protocol import safe_relative_path
    from kraken_core.safe_files import ensure_regular_directory
    from neuralimage.application.dto import MainWindowState, SettingsState
    from neuralimage.application.services.workflow_mapper import build_workflow_parameters
    from neuralimage.infrastructure.config.state_store import save_workflow_snapshot
    from neuralimage.kraken_bridge import KrakenBridgeError, _atomic_text
    from neuralimage.lib.data_interfaces import WorkMode
    from neuralimage.lib.message_bus import MessageBus
    from neuralimage.model.general_neural_handler import GeneralNeuralHandler
    import numpy as np
    import torch

    epochs, seed, overrides, validation_root = training_configuration(session.manifest.parameters)
    images, labels = session.prepare_dataset()
    defaults = dict(dataloader_num_workers=0, show_batch_preview=False)
    defaults.update(overrides)
    settings = SettingsState(**defaults)
    partition = {"validation_policy": "existing deterministic split" if settings.use_validation else "disabled"}
    if validation_root is not None:
        if session.dataset_source is not None:
            raise KrakenBridgeError(
                "External validation requires a fully staged dataset; streamed sources support split validation"
            )
        relative = safe_relative_path(validation_root)
        root = session.staging_root / "dataset"
        # Both must remain inside the original dataset, including all parents.
        from kraken_core.safe_files import contained_path

        validation_images = contained_path(root, (*relative.split("/"), "images"))
        validation_labels = contained_path(root, (*relative.split("/"), "cif"))
        ensure_regular_directory(validation_images)
        ensure_regular_directory(validation_labels)
        if validation_images.resolve() == images.resolve() or validation_labels.resolve() == labels.resolve():
            raise KrakenBridgeError("External validation must be separate from training inputs")
        if not settings.use_validation or settings.validation_source != "external":
            raise KrakenBridgeError("validation_relative_root requires explicit external validation settings")
        settings.validation_image_folder, settings.validation_label_folder = (
            str(validation_images),
            str(validation_labels),
        )
        partition = {
            "validation_policy": "disjoint external image bytes",
            **validate_external_partition(images, validation_images, recursive=settings.recursive_file_search),
        }
    elif settings.use_validation and settings.validation_source == "external":
        raise KrakenBridgeError("External validation requires validation_relative_root inside the dataset")
    main = MainWindowState(
        work_mode=WorkMode.train_only.value,
        sample_folder=str(images),
        label_folder=str(labels),
        result_folder=str(session.output_directory),
        epochs=epochs,
        ui_mode="advanced",
    )
    mode, training, recognition = build_workflow_parameters(main, settings)
    # The interactive automatic policy chooses its own guard (up to 200
    # epochs). A managed job's explicit epoch budget remains an upper bound.
    training.early_stopping.max_epochs = epochs
    training.artifact_dir = session.output_directory / "training"
    training.artifact_dir.mkdir(exist_ok=False)
    save_workflow_snapshot(
        main,
        settings,
        destination=training.artifact_dir / "neuralimage_workflow.json",
        workflow_snapshot=(mode, training, recognition),
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    errors, completed = [], []
    log_path = session.output_directory / "training-run.log"
    bus = MessageBus()

    def log(message):
        with log_path.open("a", encoding="utf-8") as output:
            output.write(str(message) + "\n")

    bus.subscribe("logging", log)

    def error(message):
        errors.append(str(message))
        log("ERROR: " + str(message))

    bus.subscribe("error", error)

    def metrics(value):
        if isinstance(value, dict) and value.get("type") == "training_completed":
            completed.append(dict(value))

    bus.subscribe("metrics", metrics)

    def question(text, *args, **kwargs):
        # Managed training cannot silently consent to an interactive decision.
        raise KrakenBridgeError("Noninteractive training requires a decision: " + str(text))

    handler = GeneralNeuralHandler(mode, question, bus, recognition, training, dataset_source=session.dataset_source)
    handler.start()
    if errors or getattr(handler, "_training_failed", False) or len(completed) != 1:
        raise KrakenBridgeError("Training did not complete successfully: " + "; ".join(errors))
    model_path = Path(completed[0]["model_path"])
    if not model_path.resolve().is_relative_to(training.artifact_dir.resolve()) or not model_path.is_file():
        raise KrakenBridgeError("Trainer did not produce its declared managed model")
    snapshot_path = training.artifact_dir / "neuralimage_workflow.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["managed_execution"] = dict(
        execution_mode="headless",
        seed=seed,
        dataset_sha256=session.manifest.inputs[0].sha256,
        plugin_runtime_torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        epoch_upper_bound=epochs,
        early_stopping_policy="automatic_dataset_size" if settings.early_stopping_enabled else "disabled",
        **partition,
    )
    _atomic_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
    return session.publish_model(model_path)
