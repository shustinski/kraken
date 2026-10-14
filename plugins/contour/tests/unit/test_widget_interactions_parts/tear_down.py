"""test_widget_interactions: tear down."""
from __future__ import annotations
from contour.widget import PolygonExtractionWidget

from test_widget_interactions_parts.oversampled_rectangle_polygon import _app


class TearDownMixin:
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self._app.processEvents()
