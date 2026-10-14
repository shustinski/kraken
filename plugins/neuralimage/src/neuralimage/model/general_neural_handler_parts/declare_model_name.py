"""general_neural_handler: declare model name."""
from __future__ import annotations
from pathlib import Path


class DeclareModelNameMixin:
    def _get_zipped_samples(self, image_folder, label_folder) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]] | tuple[list[tuple[Path, Path]], None]:
        matched_samples = self._collect_matched_samples(Path(image_folder), Path(label_folder))
        if self._need_stop or matched_samples is None:
            return [], None
        if self.tranining_parameters.validation and not self._uses_external_validation():
            return self._split_validation_samples(matched_samples)
        return matched_samples, None

    def _declare_model_name(self) -> str:
        generation = self.tranining_parameters.generation

        model_name = str(self.recognition_parameters.model)
        model_name += f'_shift_{generation.step}_'
        model_name += 'r90_' if generation.horizontal_rotation else ''
        model_name += 'r180_' if generation.vertical_rotation else ''
        model_name += 'fx_' if bool(getattr(generation, 'flip_x', False)) else ''
        model_name += 'fy_' if bool(getattr(generation, 'flip_y', False)) else ''
        model_name += f'epoch{self.tranining_parameters.epochs}'
        model_name += '.pth'

        return model_name
