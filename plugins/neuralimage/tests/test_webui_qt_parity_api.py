import os
import tempfile
import time
import zipfile
from unittest.mock import patch
from pathlib import Path

import pytest

django = pytest.importorskip("django")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "neuralimage.webui_project.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.test.utils import override_settings
from django.urls import reverse

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.lib.update_checker import UpdateClientConfig, UpdateInfo
from neuralimage.webui.services.broadcast_notifications import get_broadcast_notification_store
from neuralimage.webui.services.training_session import TrainingSessionService
import neuralimage.webui.services.training_session as training_session_module
class TestWebuiQtParityApis(TestCase):
    def setUp(self):
        presenter_stub = type(
            'PresenterStub',
            (),
            {'load_initial_states': staticmethod(lambda: (MainWindowState(), SettingsState()))},
        )()
        self.user = get_user_model().objects.create_user(username='alice', password='pw123456')
        self.other_user = get_user_model().objects.create_user(username='bob', password='pw123456')
        self.client = Client()
        self.client.force_login(self.user)
        self.service = TrainingSessionService(presenter=presenter_stub)
        self._previous_singleton = training_session_module._session_singleton
        training_session_module._session_singleton = self.service
        get_broadcast_notification_store().clear()

    def tearDown(self):
        training_session_module._session_singleton = self._previous_singleton

    def test_workflow_preset_api_returns_form_state(self):
        response = self.client.get(reverse('webui:workflow_preset_api'), data={'preset': 'conductors'})

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert 'main' in payload['state']
        assert 'settings' in payload['state']
        assert payload['preset'] == 'conductors'

    def test_ui_mode_api_persists_session_state(self):
        response = self.client.post(reverse('webui:ui_mode_api'), data={'ui_mode': 'advanced'})

        assert response.status_code == 200
        assert response.json() == {'ok': True, 'ui_mode': 'advanced'}
        assert self.client.session['webui_ui_mode'] == 'advanced'

    def test_queue_properties_api_exposes_owner_and_restore_api_blocks_foreign_user(self):
        task = self.service._processing_session.enqueue_task(
            MainWindowState(work_mode='train_only', sample_folder='sample_dir', label_folder='label_dir'),
            SettingsState(step=32),
            owner_username='alice',
            owner_display_name='Alice',
        )

        response = self.client.get(reverse('webui:queue_properties_api'), data={'task_id': task.task_id})

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert payload['task']['owner_username'] == 'alice'
        assert payload['task']['can_restore'] is True
        assert payload['task']['workflow']['main_window_state']['work_mode'] == 'train_only'

        foreign_client = Client()
        foreign_client.force_login(self.other_user)
        restore_response = foreign_client.post(
            reverse('webui:queue_restore_api'),
            data={'task_id': task.task_id},
        )

        assert restore_response.status_code == 403
        assert restore_response.json()['ok'] is False

    def test_queue_restore_api_returns_state_for_owner(self):
        task = self.service._processing_session.enqueue_task(
            MainWindowState(work_mode='recognition_only', model_path='model.pth'),
            SettingsState(batch_size=7),
            owner_username='alice',
            owner_display_name='Alice',
        )

        response = self.client.post(reverse('webui:queue_restore_api'), data={'task_id': task.task_id})

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert payload['state']['main']['work_mode'] == 'recognition_only'
        assert payload['state']['settings']['batch_size'] == 7

    def test_help_and_changelog_content_apis_return_markdown(self):
        help_response = self.client.get(reverse('webui:help_content_api'))
        changelog_response = self.client.get(reverse('webui:changelog_content_api'))

        assert help_response.status_code == 200
        assert help_response.json()['ok'] is True
        assert help_response.json()['content']
        assert changelog_response.status_code == 200
        assert changelog_response.json()['ok'] is True
        assert changelog_response.json()['content']

    def test_dashboard_contains_qt_style_action_controls(self):
        response = self.client.get(reverse('webui:dashboard'))
        content = response.content.decode('utf-8')

        assert response.status_code == 200
        assert 'qt-menubar' in content
        assert 'data-menu-action="import-workflow"' in content
        assert 'data-menu-action="settings-train"' in content
        assert 'data-menu-action="toggle-metrics"' in content
        assert 'data-menu-action="augmentation-preview"' in content
        assert 'data-menu-action="rare-patch-editor"' in content
        assert 'data-menu-action="developer"' in content
        assert 'sample-count-value' in content
        assert 'toggle-metrics-btn' not in content
        assert 'toggle-preview-btn' not in content
        assert 'release-memory-btn' not in content
        assert 'reset-defaults-btn' in content
        assert content.index('id="settings-section-preprocessing"') < content.index('id="reset-defaults-btn"')
        assert 'import-workflow-btn' not in content
        assert 'open-help-btn' not in content
        assert 'open-changelog-btn' not in content
        assert 'check-updates-btn' not in content
        assert 'data-theme="light"' not in content
        assert 'settings-nav-btn' in content
        assert content.index('id="settings-section-sample"') < content.index('name="main-epochs"')
        assert content.index('id="settings-section-spatial"') < content.index('id="settings-section-rare"')
        assert content.index('id="settings-section-rare"') < content.index('id="settings-section-optimizer"')
        assert content.index('id="settings-section-optimizer"') < content.index('id="settings-section-loss"')

    def test_pick_path_api_accepts_remote_folder_upload_and_returns_server_folder(self):
        with tempfile.TemporaryDirectory() as temp_root, override_settings(WEBUI_UPLOAD_ROOT=Path(temp_root)):
            response = self.client.post(
                reverse('webui:pick_path_api'),
                data={
                    'kind': 'folder',
                    'target': 'main-sample_folder',
                    'files': [
                        SimpleUploadedFile('sample_a.png', b'a'),
                        SimpleUploadedFile('sample_b.png', b'b'),
                    ],
                    'relative_paths': ['chip/sample_a.png', 'chip/nested/sample_b.png'],
                },
                REMOTE_ADDR='10.0.0.2',
                HTTP_HOST='192.168.1.10:8000',
            )

            assert response.status_code == 200
            payload = response.json()
            server_folder = Path(payload['path'])
            assert payload['ok'] is True
            assert payload['count'] == 2
            assert server_folder.is_dir()
            assert (server_folder / 'chip' / 'sample_a.png').read_bytes() == b'a'
            assert (server_folder / 'chip' / 'nested' / 'sample_b.png').read_bytes() == b'b'

    def test_pick_path_api_accepts_remote_model_upload_and_returns_server_file(self):
        with tempfile.TemporaryDirectory() as temp_root, override_settings(WEBUI_UPLOAD_ROOT=Path(temp_root)):
            response = self.client.post(
                reverse('webui:pick_path_api'),
                data={
                    'kind': 'file',
                    'filter': 'model',
                    'target': 'main-model_path',
                    'files': [SimpleUploadedFile('model.pth', b'model-bytes')],
                    'relative_paths': ['model.pth'],
                },
                REMOTE_ADDR='10.0.0.2',
                HTTP_HOST='192.168.1.10:8000',
            )

            assert response.status_code == 200
            payload = response.json()
            server_file = Path(payload['path'])
            assert payload['ok'] is True
            assert payload['count'] == 1
            assert server_file.is_file()
            assert server_file.read_bytes() == b'model-bytes'

    def test_pick_path_api_creates_managed_result_folder_for_remote_client(self):
        with tempfile.TemporaryDirectory() as temp_root, override_settings(WEBUI_UPLOAD_ROOT=Path(temp_root)):
            response = self.client.post(
                reverse('webui:pick_path_api'),
                data={'kind': 'folder', 'target': 'main-result_folder'},
                REMOTE_ADDR='10.0.0.2',
                HTTP_HOST='192.168.1.10:8000',
            )

            assert response.status_code == 200
            payload = response.json()
            server_folder = Path(payload['path'])
            assert payload == {'ok': True, 'path': str(server_folder), 'count': 0}
            assert server_folder.is_dir()

    def test_pick_path_api_unpacks_zip_folder_upload(self):
        with tempfile.TemporaryDirectory() as temp_root:
            archive_path = Path(temp_root) / 'dataset.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('images/a.png', b'a')
                archive.writestr('images/b.png', b'b')

            with override_settings(WEBUI_UPLOAD_ROOT=Path(temp_root) / 'uploads'):
                response = self.client.post(
                    reverse('webui:pick_path_api'),
                    data={
                        'kind': 'folder',
                        'target': 'main-sample_folder',
                        'files': [SimpleUploadedFile('dataset.zip', archive_path.read_bytes())],
                    },
                    REMOTE_ADDR='10.0.0.2',
                    HTTP_HOST='192.168.1.10:8000',
                )

                assert response.status_code == 200
                payload = response.json()
                server_folder = Path(payload['path'])
                assert payload['ok'] is True
                assert payload['count'] == 2
                assert (server_folder / 'images' / 'a.png').read_bytes() == b'a'
                assert (server_folder / 'images' / 'b.png').read_bytes() == b'b'

    def test_upload_root_cleanup_removes_stale_children(self):
        from neuralimage.webui.views import _webui_upload_root

        with tempfile.TemporaryDirectory() as temp_root, override_settings(
            WEBUI_UPLOAD_ROOT=Path(temp_root),
            WEBUI_UPLOAD_MAX_AGE_SECONDS=1,
        ):
            stale = Path(temp_root) / 'stale'
            fresh = Path(temp_root) / 'fresh'
            stale.mkdir()
            fresh.mkdir()
            old_time = time.time() - 3600
            os.utime(stale, (old_time, old_time))

            root = _webui_upload_root()

            assert root == Path(temp_root)
            assert not stale.exists()
            assert fresh.exists()

    def test_streaming_recognition_api_returns_result_zip_without_persisting_source(self):
        with tempfile.TemporaryDirectory() as temp_root:
            model_path = Path(temp_root) / 'model.pth'
            model_path.write_bytes(b'model')

            class FakeRecognizer:
                def __init__(self, recognition_parameters, message_bus):
                    self._parameters = recognition_parameters

                def run(self, multithreading=None):
                    assert Path(self._parameters.source_folder).is_dir()
                    assert any(Path(self._parameters.source_folder).rglob('source.png'))
                    Path(self._parameters.result_folder, 'mask.png').write_bytes(b'mask')

            with override_settings(WEBUI_UPLOAD_ROOT=Path(temp_root) / 'uploads'), patch(
                'model.NeuralNetwork.model_train_and_recognition.NeuralRecognizer',
                FakeRecognizer,
            ):
                response = self.client.post(
                    reverse('webui:streaming_recognition_api'),
                    data={
                        'main-work_mode': 'recognition_only',
                        'main-source_folder': 'browser-stream://1-files',
                        'main-result_folder': 'browser-result://results',
                        'main-model_path': str(model_path),
                        'main-epochs': '1',
                        'settings-step': '100',
                        'settings-sample_x': '256',
                        'settings-sample_y': '256',
                        'settings-model': 'M 720k',
                        'settings-color_mode': 'RGB',
                        'settings-sample_cut_mode': 'online',
                        'settings-batch_size': '1',
                        'settings-overlap': '16',
                        'settings-log_update_frequency': '0',
                        'settings-optimizer_name': 'adam',
                        'settings-mixed_precision': 'bf16',
                        'settings-loss_function': 'bce',
                        'settings-learning_rate': '0.001',
                        'settings-weight_decay': '0.0',
                        'settings-warmup_epochs': '1',
                        'settings-warmup_start_factor': '0.1',
                        'settings-early_stopping_patience': '10',
                        'settings-early_stopping_min_delta': '0.0',
                        'source_files': [SimpleUploadedFile('source.png', b'image')],
                        'source_relative_paths': ['source.png'],
                    },
                    REMOTE_ADDR='10.0.0.2',
                    HTTP_HOST='192.168.1.10:8000',
                )

            assert response.status_code == 200
            payload = b''.join(response.streaming_content)
            result_zip = Path(temp_root) / 'result.zip'
            result_zip.write_bytes(payload)
            with zipfile.ZipFile(result_zip) as archive:
                assert archive.read('mask.png') == b'mask'

    def test_dashboard_uses_russian_epochs_label_in_settings_panel(self):
        response = self.client.get(reverse('webui:dashboard'), data={'lang': 'ru'})
        content = response.content.decode('utf-8')

        assert response.status_code == 200
        assert 'Epoch count' not in content
        assert 'Эпох обучения' in content

    def test_sample_count_api_returns_zero_for_missing_folder(self):
        response = self.client.post(
            reverse('webui:sample_count_api'),
            data={'main-sample_folder': '', 'settings-sample_x': '256', 'settings-sample_y': '256'},
        )

        assert response.status_code == 200
        assert response.json() == {'ok': True, 'count': 0}

    def test_sample_count_api_calculates_total(self):
        with patch('webui.views.SampleWorker.collect_image_paths', return_value=[Path('a.png'), Path('b.png')]), patch(
            'webui.views.SampleWorker.collect_image_sizes',
            return_value=[(512, 512), (512, 512)],
        ), patch('pathlib.Path.is_dir', return_value=True):
            response = self.client.post(
                reverse('webui:sample_count_api'),
                data={
                    'main-sample_folder': 'D:/dataset',
                    'settings-sample_x': '256',
                    'settings-sample_y': '256',
                    'settings-step': '256',
                    'settings-vertical_rotation': 'on',
                    'settings-horizontal_rotation': '',
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert payload['count'] > 0

    def test_release_memory_api_succeeds_without_cuda(self):
        fake_torch = type(
            'FakeTorch',
            (),
            {'cuda': type('FakeCuda', (), {'is_available': staticmethod(lambda: False)})()},
        )
        with patch.dict('sys.modules', {'torch': fake_torch}):
            response = self.client.post(reverse('webui:release_memory_api'))

        assert response.status_code == 200
        assert response.json() == {'ok': True, 'cuda_available': False}

    def test_reset_defaults_api_returns_default_form_state(self):
        response = self.client.post(reverse('webui:reset_defaults_api'))

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert payload['state']['main']['epochs'] == 20
        assert payload['state']['settings']['model'] == 'M 720k'

    def test_tool_status_api_returns_controlled_status(self):
        response = self.client.get(reverse('webui:tool_status_api'), data={'tool': 'augmentation_preview'})

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert payload['tool'] == 'augmentation_preview'
        assert payload['message']

    def test_update_info_api_returns_configured_payload(self):
        with patch(
            'webui.views.load_update_client_config',
            return_value=UpdateClientConfig(
                manifest_urls=(('stable', 'file:///tmp/update.json'), ('beta', 'file:///tmp/update-beta.json')),
                default_channel='stable',
            ),
        ), patch(
            'webui.views.fetch_update_info',
            return_value=UpdateInfo(
                version='9.9.9',
                download_url='https://example.invalid/download',
                release_notes='Notes',
                channel='beta',
            ),
        ):
            response = self.client.get(reverse('webui:update_info_api'), data={'channel': 'beta'})

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert payload['configured'] is True
        assert payload['selected_channel'] == 'beta'
        assert payload['new_version'] == '9.9.9'

    def test_broadcast_notification_api_accepts_admin_token_and_status_returns_it(self):
        with patch.dict(os.environ, {'NEURALIMAGE_WEBUI_ADMIN_TOKEN': 'secret-token'}):
            response = self.client.post(
                reverse('webui:broadcast_notification_api'),
                data={'message': 'Maintenance in 10 minutes', 'created_by': 'Qt Developer'},
                HTTP_X_NEURALIMAGE_ADMIN_TOKEN='secret-token',
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload['ok'] is True
        assert payload['notification']['message'] == 'Maintenance in 10 minutes'

        status_response = self.client.get(reverse('webui:status_api'), data={'notification_after': 0})
        status_payload = status_response.json()

        assert status_response.status_code == 200
        assert status_payload['notifications'][-1]['message'] == 'Maintenance in 10 minutes'
        assert status_payload['last_notification_id'] >= payload['notification']['id']

    def test_broadcast_notification_api_rejects_missing_token_for_regular_user(self):
        response = self.client.post(
            reverse('webui:broadcast_notification_api'),
            data={'message': 'forbidden'},
        )

        assert response.status_code == 403
        assert response.json()['ok'] is False
