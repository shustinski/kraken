"""manager_app: representation file count."""
from __future__ import annotations
from pathlib import Path

from kraken_hub.presentation.qt.parts.event_matches_node import _count_regular_files


class RepresentationFileCountMixin:
    def _representation_file_count(self, layer_id: str, representation) -> int | None:
        try:
            cells = self.service.frame_cells(
                self._project_id,
                layer_id,
                representation.id,
            )
        except (OSError, ValueError):
            cells = ()
        if cells:
            return len({str(item.frame_id) for item in cells})
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        if (
            binding is not None
            and representation.source
            and str(Path(representation.source)).casefold()
            == str(Path(binding.image_directory)).casefold()
            and binding.frame_positions
        ):
            return len(binding.frame_positions)
        return _count_regular_files((representation.source,))
