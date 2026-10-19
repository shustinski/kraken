"""Layer stitching files use the existing immutable artifact version chains."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid5

from .artifacts import ArtifactScope, ArtifactSeries, ArtifactVersion
from .common import ArtifactSeriesId, DomainValidationError


class StitchingKind(StrEnum):
    INTRA_LAYER = "intra_layer"
    INTER_LAYER = "inter_layer"

    @property
    def title(self) -> str:
        return "Внутрислойная сшивка" if self is StitchingKind.INTRA_LAYER else "Межслойная сшивка"


def stitching_series_id(layer_id: object, kind: StitchingKind | str) -> ArtifactSeriesId:
    return ArtifactSeriesId(str(uuid5(UUID(str(layer_id)), f"stitching:{StitchingKind(kind).value}")))


def validate_stitching_file(
    project_id: object,
    layer_id: object,
    kind: StitchingKind | str,
    version: ArtifactVersion | None,
    series: ArtifactSeries | None,
) -> None:
    if (
        version is None
        or version.blob is None
        or series is None
        or series.archived
        or series.scope is not ArtifactScope.LAYER_ATTACHMENT
        or str(series.project_id) != str(project_id)
        or str(series.layer_id) != str(layer_id)
        or series.id != stitching_series_id(layer_id, kind)
        or version.series_id != series.id
    ):
        raise DomainValidationError("Сначала загрузите файл соответствующего вида сшивки в этот слой")
