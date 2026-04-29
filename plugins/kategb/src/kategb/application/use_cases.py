from __future__ import annotations

from pathlib import Path
from typing import Protocol

from kategb.application.dto import AnalyzeVerificationRequest, GenerateManifestRequest, SampleRequest
from kategb.domain.cipher import decode_string, encode_string
from kategb.domain.frames import generate_sample_frames, parse_frame_range
from kategb.domain.models import CrystalInfo, IncorrectVector, SampleGenerationConfig, VerificationManifest
from kategb.domain.quality import calculate_author_results


class CrystalInfoReader(Protocol):
    def read(self, path: Path) -> CrystalInfo: ...


class IncorrectVectorReader(Protocol):
    def read(self, path: Path) -> dict[int, IncorrectVector]: ...


class BuildSample:
    def execute(self, crystal_info: CrystalInfo, request: SampleRequest) -> tuple[int, ...]:
        layer = crystal_info.layers[request.layer_name]
        frame_range = parse_frame_range(request.frame_range_text) if request.frame_range_text.strip() else None
        config = SampleGenerationConfig(
            layer_name=request.layer_name,
            authors=request.authors,
            percent_per_author=request.percent_per_author,
            frame_range=frame_range,
            selection_mode=request.selection_mode,
            random_seed=request.random_seed,
        )
        return generate_sample_frames(layer, config)


class GenerateVerificationManifest:
    def execute(self, request: GenerateManifestRequest) -> Path:
        if not request.frames:
            raise ValueError("Нельзя сохранить файл проверки без кадров.")
        request.output_folder.mkdir(parents=True, exist_ok=True)
        manifest = VerificationManifest(
            vector_folder=request.vector_folder,
            layer_name=request.layer_name,
            frame_range=request.frame_range_text,
            check_name=request.check_name,
            selection_mode=request.selection_mode,
            frames=request.frames,
        )
        path = request.output_folder / f"{request.check_name}.txt"
        lines = [
            manifest.vector_folder,
            manifest.layer_name,
            manifest.frame_range,
            manifest.check_name,
            manifest.selection_mode,
            *(str(frame) for frame in manifest.frames),
        ]
        path.write_text("\n".join(encode_string(line, request.encryption_key) for line in lines) + "\n", encoding="utf-8")
        return path


class ReadVerificationManifest:
    def execute(self, path: Path, encryption_key: str) -> VerificationManifest:
        raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(raw_lines) < 6:
            raise ValueError("Файл проверки неполный.")
        decoded = [decode_string(line, encryption_key) for line in raw_lines]
        selection_mode = decoded[4]
        if selection_mode not in {"all", "area"}:
            raise ValueError("Неверный ключ шифрования или режим диапазона в файле проверки.")
        return VerificationManifest(
            vector_folder=decoded[0],
            layer_name=decoded[1],
            frame_range=decoded[2],
            check_name=decoded[3],
            selection_mode=selection_mode,
            frames=tuple(int(item) for item in decoded[5:]),
        )


class AnalyzeVerification:
    def __init__(self, crystal_reader: CrystalInfoReader, incorrect_reader: IncorrectVectorReader) -> None:
        self._crystal_reader = crystal_reader
        self._incorrect_reader = incorrect_reader
        self._read_manifest = ReadVerificationManifest()

    def execute(self, request: AnalyzeVerificationRequest):
        if request.markup_path is None:
            raise ValueError("Для сопоставления кадров с исполнителями нужен файл разметки.")
        manifest = self._read_manifest.execute(request.manifest_path, request.encryption_key)
        crystal = self._crystal_reader.read(request.markup_path)
        layer_name = request.layer_name or manifest.layer_name
        layer = crystal.layers[layer_name]
        incorrect = self._incorrect_reader.read(request.incorrect_xml_path)
        checked_numbers = tuple(incorrect)
        return calculate_author_results(layer, manifest.frames, checked_numbers, incorrect)
