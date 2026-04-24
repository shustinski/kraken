# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import os

from PyInstaller.utils import hooks


DJANGO_PROJECT_PACKAGE = 'webui_project'
DJANGO_SETTINGS_MODULE = 'webui_project.settings'


datas, binaries, hiddenimports = hooks.collect_all('django', on_error='ignore')

hiddenimports += [
    DJANGO_SETTINGS_MODULE,
    f'{DJANGO_PROJECT_PACKAGE}.urls',
    f'{DJANGO_PROJECT_PACKAGE}.wsgi',
    f'{DJANGO_PROJECT_PACKAGE}.asgi',
    'http.cookies',
    'html.parser',
]

try:
    installed_apps = hooks.get_module_attribute(DJANGO_SETTINGS_MODULE, 'INSTALLED_APPS')
except Exception:
    installed_apps = []

migration_modules = [
    'django.conf.app_template.migrations',
    'django.contrib.admin.migrations',
    'django.contrib.auth.migrations',
    'django.contrib.contenttypes.migrations',
    'django.contrib.flatpages.migrations',
    'django.contrib.redirects.migrations',
    'django.contrib.sessions.migrations',
    'django.contrib.sites.migrations',
]
migration_modules.extend({f'{app}.migrations' for app in installed_apps})

for module_name in migration_modules:
    top_module, bundle_name = module_name.split('.', 1)
    try:
        module_dir = os.path.dirname(hooks.get_module_file_attribute(top_module))
    except Exception:
        continue
    bundle_dir = bundle_name.replace('.', os.sep)
    pattern = os.path.join(module_dir, bundle_dir, '*.py')
    for source_path in glob.glob(pattern):
        datas.append((source_path, os.path.join(top_module, bundle_dir)))

datas += hooks.collect_data_files(DJANGO_PROJECT_PACKAGE)
