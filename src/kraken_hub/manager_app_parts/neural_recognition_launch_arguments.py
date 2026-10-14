"""manager_app: neural recognition launch arguments."""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from uuid import uuid4
from kraken_core.plugin_protocol import (
    PluginFrameInput,
    PluginJobManifest,
)
from kraken_manager.domain.project import GridOrientation as DomainOrientation
from kraken_manager.workspace import DerivedRunKind


class NeuralRecognitionLaunchArgumentsMixin:
    def _neural_recognition_launch_arguments(
        self,
        *,
        layer_id: str,
        source_representation_id: str,
        stage_root: Path,
        staged_model,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        project = self.service.get_project(self._project_id)
        layer = next(
            (
                value
                for value in self.service.list_layers(self._project_id)
                if str(value.id) == str(layer_id)
            ),
            None,
        )
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        representation = next(
            (
                value
                for value in self.service.list_representations(
                    self._project_id, layer_id
                )
                if str(value.id) == str(source_representation_id)
            ),
            None,
        )
        if project is None or layer is None or binding is None or representation is None:
            raise ValueError("проект, слой или исходные изображения недоступны")
        image_directory = Path(str(representation.source or binding.image_directory))
        if not image_directory.is_dir():
            raise ValueError("папка исходных изображений недоступна")
        input_stage = stage_root / "inputs"
        input_stage.mkdir(exist_ok=False)
        inputs: list[PluginFrameInput] = []
        frame_positions_by_id: dict[str, int] = {}
        image_paths = sorted(
            (
                path
                for path in image_directory.iterdir()
                if path.is_file()
                and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp"}
            ),
            key=self.service._natural_path_key,
        )
        for path in image_paths:
            position = binding.frame_positions.get(path.name)
            if position is None:
                continue
            visual_row, column = divmod(position - 1, project.width)
            x = column + 1
            y = (
                visual_row + 1
                if project.orientation is DomainOrientation.Y_DOWN
                else project.height - visual_row
            )
            if path.suffix.casefold() == ".bmp":
                from PIL import Image, ImageOps

                destination = input_stage / f"{path.stem}.png"
                with Image.open(path) as opened:
                    ImageOps.exif_transpose(opened).save(destination, format="PNG")
            else:
                destination = input_stage / path.name
                shutil.copy2(path, destination)
            digest_builder = hashlib.sha256()
            with destination.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest_builder.update(chunk)
            digest = digest_builder.hexdigest()
            frame_id = str(project.frame_id_at(x, y))
            # Layer bindings use Kraken's one-based matrix slots internally,
            # while microscope and plugin filenames are always zero-based.
            frame_positions_by_id[frame_id] = position - 1
            inputs.append(
                PluginFrameInput(
                    frame_id=frame_id,
                    x=x,
                    y=y,
                    artifact_version_id=str(uuid4()),
                    sha256=digest,
                    media_type=(
                        "image/jpeg"
                        if destination.suffix.casefold() in {".jpg", ".jpeg"}
                        else "image/png"
                        if destination.suffix.casefold() == ".png"
                        else "image/bmp"
                    ),
                    relative_path=f"inputs/{destination.name}",
                )
            )
        if not inputs:
            raise ValueError("не найдено изображений с валидной нумерацией кадров")
        operation = self._action_requirement("recognize")[1]
        run = self.service.begin_derived_run(
            project_id=project.id,
            layer_id=layer.id,
            layer_name=layer.name,
            kind=DerivedRunKind.RESULT,
            plugin_id="neuralimage",
            operation=operation,
            principal=self.session.principal,
        )
        job_id = str(uuid4())
        job_manifest = stage_root / "job.json"
        result_manifest = stage_root / "result.json"
        manifest = PluginJobManifest(
            job_id=job_id,
            operation=operation,
            project_id=str(project.id),
            layer_id=str(layer.id),
            actor_id=str(self.session.principal.id),
            target_representation_id=str(representation.id),
            inputs=tuple(inputs),
            parameters={
                "model_relative_path": staged_model.relative_path,
                "model_sha256": staged_model.used_sha256,
                "model_version": f"sha256:{staged_model.used_sha256[:12]}",
                "lossless_binary_png": True,
                "frame_positions": frame_positions_by_id,
            },
        )
        with job_manifest.open("x", encoding="utf-8") as stream:
            stream.write(manifest.to_json())
        return (
            (
                "--kraken-job-manifest",
                str(job_manifest),
                "--kraken-result-manifest",
                str(result_manifest),
                "--kraken-staging-root",
                str(stage_root),
            ),
            {
                "workspace_run_id": run.run_id,
                "agent_staging_root": str(stage_root),
                "agent_result_manifest": str(result_manifest),
                "frame_count": len(inputs),
            },
        )
