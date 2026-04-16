from django.urls import path

from . import views

app_name = 'webui'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('auth/ldap/login/', views.auth_ldap_login, name='auth_ldap_login'),
    path('auth/superuser/login/', views.auth_superuser_login, name='auth_superuser_login'),
    path('auth/logout/', views.auth_logout, name='auth_logout'),
    path('start/', views.start_processing, name='start_processing'),
    path('stop/', views.stop_processing, name='stop_processing'),
    path('api/status/', views.status_api, name='status_api'),
    path('api/broadcast/', views.broadcast_notification_api, name='broadcast_notification_api'),
    path('api/queue/remove/', views.queue_remove_api, name='queue_remove_api'),
    path('api/queue/pause-toggle/', views.queue_pause_toggle_api, name='queue_pause_toggle_api'),
    path('api/queue/properties/', views.queue_properties_api, name='queue_properties_api'),
    path('api/queue/restore/', views.queue_restore_api, name='queue_restore_api'),
    path('api/workflow/import/', views.workflow_import_api, name='workflow_import_api'),
    path('api/workflow/preset/', views.workflow_preset_api, name='workflow_preset_api'),
    path('api/recognition/stream/', views.streaming_recognition_api, name='streaming_recognition_api'),
    path('api/help/', views.help_content_api, name='help_content_api'),
    path('api/changelog/', views.changelog_content_api, name='changelog_content_api'),
    path('api/update-info/', views.update_info_api, name='update_info_api'),
    path('api/ui-mode/', views.ui_mode_api, name='ui_mode_api'),
    path('api/sample-count/', views.sample_count_api, name='sample_count_api'),
    path('api/release-memory/', views.release_memory_api, name='release_memory_api'),
    path('api/reset-defaults/', views.reset_defaults_api, name='reset_defaults_api'),
    path('api/tool-status/', views.tool_status_api, name='tool_status_api'),
    path('api/pick-path/', views.pick_path_api, name='pick_path_api'),
]
