"""Local adapter for shared published-analysis queries."""
from kraken_manager.application.frame_queries import latest_karakal_analysis, karakal_analysis_from_event


class LatestKarakalAnalysisMixin:
    def latest_karakal_analysis(self, project_id, layer_id, *, as_of=None):
        return latest_karakal_analysis(self.history, project_id, layer_id, as_of=as_of)

    _karakal_run_from_event = staticmethod(karakal_analysis_from_event)
