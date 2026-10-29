"""Indexed frame-row adapter for stores exposing the filesystem matrix index."""


class ProjectionFrameIndex:
    def __init__(self, projection):
        self._projection = projection

    def rows(self, project_id, layer_id, representation_id, bounds):
        return self._projection.matrix_frame_rows(project_id, layer_id, representation_id, bounds)
