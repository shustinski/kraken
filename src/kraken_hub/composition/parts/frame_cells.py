"""Local adapter for shared frame read-model policies."""
from kraken_manager.application.frame_queries import FrameQueryService
from kraken_manager.infrastructure.frame_index import ProjectionFrameIndex


class FrameCellsMixin:
    def _frame_queries(self, project_id):
        projection = self._projection(project_id)
        return FrameQueryService(projection, history=self.history, performers=self.list_performers,
                                 frame_index=ProjectionFrameIndex(projection))

    def frame_cells(self, project_id, layer_id, representation_id, *, as_of=None, bounds=None):
        return self._frame_queries(project_id).frame_cells(project_id, layer_id, representation_id,
                                                          as_of=as_of, bounds=bounds)
