"""composition: reorder layers."""
from __future__ import annotations
import re
from collections.abc import Iterable
from pathlib import Path
from kraken_manager.application.dto import (
    CommandContext,
    ReorderLayersCommand,
)
from kraken_manager.application.lifecycle import (
    ReorderLayersHandler,
)
from kraken_manager.domain.common import (
    LayerId,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
)


class ReorderLayersMixin:
    @staticmethod
    def _natural_path_key(path: Path) -> tuple[object, ...]:
        return tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        )

    def reorder_layers(
        self,
        *,
        principal: Principal,
        project: Project,
        layers: Iterable[Layer],
        layer_ids: Iterable[LayerId | str],
        idempotency_key: str,
    ) -> tuple[Layer, ...]:
        current = tuple(layers)
        return ReorderLayersHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ReorderLayersCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                layer_ids=tuple(LayerId(str(value)) for value in layer_ids),
                expected_revisions=tuple((layer.id, layer.revision) for layer in current),
            )
        )
