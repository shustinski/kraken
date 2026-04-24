import os
from unittest.mock import patch

import pytest

django = pytest.importorskip("django")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webui_project.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse


class TestWebuiLocalSuperuserAuth(TestCase):
    def test_ldap_login_endpoint_creates_and_logs_in_user(self):
        user_model = get_user_model()
        client = Client()

        with patch.dict(os.environ, {'NEURALIMAGE_LDAP_SERVER_URI': 'ldap://ldap.example.test'}), patch(
            'webui.views._ldap_authenticate',
            return_value={'username': 'alice', 'display_name': 'Alice Engineer', 'email': 'alice@example.test'},
        ):
            response = client.post(
                reverse('webui:auth_ldap_login'),
                data={'username': 'alice', 'password': 'secret'},
            )

        assert response.status_code == 302
        assert response.url == reverse('webui:dashboard')
        user = user_model.objects.get(username='alice')
        assert user.first_name == 'Alice Engineer'
        assert user.email == 'alice@example.test'
        assert '_auth_user_id' in client.session

    def test_ldap_login_endpoint_rejects_when_not_configured(self):
        client = Client()

        with patch.dict(os.environ, {}, clear=True):
            response = client.post(
                reverse('webui:auth_ldap_login'),
                data={'username': 'alice', 'password': 'secret'},
            )

        assert response.status_code == 302
        assert response.url == reverse('webui:dashboard')
        assert '_auth_user_id' not in client.session

    def test_superuser_login_endpoint_creates_and_logs_in_local_admin(self):
        user_model = get_user_model()
        client = Client()

        response = client.post(
            reverse('webui:auth_superuser_login'),
            data={'username': 'not_admin', 'password': 'Aa123456'},
        )

        assert response.status_code == 302
        assert response.url == reverse('webui:dashboard')
        user = user_model.objects.get(username='not_admin')
        assert user.is_staff is True
        assert user.is_superuser is True
        assert user.check_password('Aa123456') is True
        assert '_auth_user_id' in client.session

    def test_dashboard_login_page_contains_superuser_button(self):
        with patch.dict(os.environ, {'NEURALIMAGE_LDAP_SERVER_URI': 'ldap://ldap.example.test'}):
            response = Client().get(reverse('webui:dashboard'))

        assert response.status_code == 401
        content = response.content.decode('utf-8')
        assert reverse('webui:auth_ldap_login') in content
        assert reverse('webui:auth_superuser_login') in content
        assert 'GitLab' not in content

    def test_superuser_login_endpoint_rejects_invalid_password(self):
        client = Client()

        response = client.post(
            reverse('webui:auth_superuser_login'),
            data={'username': 'not_admin', 'password': 'wrong'},
        )

        assert response.status_code == 302
        assert response.url == reverse('webui:dashboard')
        assert '_auth_user_id' not in client.session
