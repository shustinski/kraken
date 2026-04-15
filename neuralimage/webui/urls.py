from django.urls import path

from . import views

app_name = 'webui'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('start/', views.start_processing, name='start_processing'),
    path('stop/', views.stop_processing, name='stop_processing'),
    path('api/status/', views.status_api, name='status_api'),
    path('api/queue/remove/', views.queue_remove_api, name='queue_remove_api'),
    path('api/queue/pause-toggle/', views.queue_pause_toggle_api, name='queue_pause_toggle_api'),
    path('api/pick-path/', views.pick_path_api, name='pick_path_api'),
]
