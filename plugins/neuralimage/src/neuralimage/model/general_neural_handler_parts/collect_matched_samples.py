"""general_neural_handler: collect matched samples."""
from __future__ import annotations
from pathlib import Path
from neuralimage.lib.file_func import filter_images


class CollectMatchedSamplesMixin:
    def _collect_matched_samples(
        self,
        image_folder: Path,
        label_folder: Path,
    ) -> list[tuple[Path, Path]] | None:
        if self.dataset_source is not None:
            if image_folder != self.dataset_source.image_root or label_folder != self.dataset_source.label_root:
                raise ValueError('The selected dataset is not registered in the server input source')
            return list(self.dataset_source.sample_pairs)
        recursive = bool(getattr(self.tranining_parameters, 'recursive_file_search', False))
        image_files = sorted(filter_images(image_folder, recursive=recursive))
        label_files = sorted(filter_images(label_folder, recursive=recursive))

        def _sample_key(file: Path, root: Path) -> str:
            if not recursive:
                return file.stem
            try:
                relative = file.relative_to(root)
            except ValueError:
                return file.stem
            return relative.with_suffix('').as_posix()

        def _build_file_map(files: list[Path], kind: str, root: Path) -> dict[str, Path]:
            result: dict[str, Path] = {}
            duplicates: list[str] = []
            for file in files:
                stem = _sample_key(file, root)
                if stem in result:
                    duplicates.append(stem)
                    continue
                result[stem] = file
            if duplicates:
                unique_duplicates = ', '.join(sorted(set(duplicates))[:10])
                self.message_bus.publish(
                    'error',
                    f'Duplicate {kind} stems detected: {unique_duplicates}. '
                    'Ensure unique base file names before training.',
                )
                self._need_stop = True
            return result

        image_map = _build_file_map(image_files, 'image', image_folder)
        label_map = _build_file_map(label_files, 'label', label_folder)
        if self._need_stop:
            return None

        image_stems = set(image_map.keys())
        label_stems = set(label_map.keys())
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        common_stems = sorted(image_stems & label_stems)
        if (missing_labels or missing_images) and common_stems:
            missing_labels_preview = ', '.join(missing_labels[:10]) if missing_labels else '-'
            missing_images_preview = ', '.join(missing_images[:10]) if missing_images else '-'
            pair_word = 'pair' if len(common_stems) == 1 else 'pairs'
            self.message_bus.publish(
                'warning',
                (
                    'Image/label mismatch detected. '
                    f'Missing labels for images: {missing_labels_preview}. '
                    f'Missing images for labels: {missing_images_preview}. '
                    f'Training will continue with {len(common_stems)} matched {pair_word}.'
                ),
            )

        zipped_images = [(image_map[stem], label_map[stem]) for stem in common_stems]
        if not zipped_images:
            self.message_bus.publish('error', 'No matched image/label pairs found in the selected dataset.')
            self._need_stop = True
            return None
        return zipped_images
