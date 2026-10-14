"""main_presenter: resolve latest release."""
from __future__ import annotations
from neuralimage.lib.update_checker import (
    ReleaseInfo,
    UpdateInfo,
    launch_update_installer,
)
from neuralimage.lib.version import APP_VERSION
import neuralimage.presenter.update_flow as update_flow
from neuralimage.presenter.background_workers import (
    AppUpdateDownloadThread as PresenterAppUpdateDownloadThread,
)


class ResolveLatestReleaseMixin:
    def _confirm_release_install(self, release: ReleaseInfo) -> None:
        update_flow.confirm_release_install(self, release, app_version=APP_VERSION)

    @staticmethod
    def _resolve_latest_release(update_info: UpdateInfo) -> ReleaseInfo:
        for release in update_info.releases:
            if release.version == update_info.version:
                return release
        return ReleaseInfo(
            version=update_info.version,
            download_url=update_info.download_url,
            notes=update_info.release_notes,
        )

    def _start_update_download(self, release: ReleaseInfo) -> None:
        update_flow.start_update_download(
            self,
            release,
            update_download_thread_cls=PresenterAppUpdateDownloadThread,
        )

    def _clear_update_download_thread(self) -> None:
        update_flow.clear_update_download_thread(self)

    def _on_update_download_finished(self, installer_path: str, version: str) -> None:
        update_flow.on_update_download_finished(
            self,
            installer_path,
            version,
            launch_update_installer_fn=launch_update_installer,
        )

    def _on_update_download_failed(self, error_message: str) -> None:
        update_flow.on_update_download_failed(self, error_message)
